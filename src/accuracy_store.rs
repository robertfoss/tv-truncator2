//! SQLite persistence for detection accuracy runs (MEMA-26).
//!
//! Use this to record metrics while iterating on algorithms or deduplication strategies, then
//! compare runs locally. The schema is versioned in-repo; default path for manual snapshots is
//! `tests/fixtures/accuracy_runs.sqlite` (gitignored — generate locally).

use crate::accuracy::DetectionAccuracyMetrics;
use crate::Result;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS accuracy_run (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  git_revision TEXT,
  label TEXT NOT NULL,
  similarity_algorithm TEXT NOT NULL,
  audio_algorithm TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS accuracy_fixture_result (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES accuracy_run(id) ON DELETE CASCADE,
  fixture_path TEXT NOT NULL,
  precision REAL NOT NULL,
  recall REAL NOT NULL,
  f1_score REAL NOT NULL,
  timing_mean_abs_error_ms REAL NOT NULL,
  true_positives INTEGER NOT NULL,
  false_positives INTEGER NOT NULL,
  false_negatives INTEGER NOT NULL,
  segments_detected INTEGER NOT NULL,
  UNIQUE(run_id, fixture_path)
);

CREATE INDEX IF NOT EXISTS idx_accuracy_fixture_result_run
  ON accuracy_fixture_result(run_id);
"#;

/// Metadata for one stored benchmark run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccuracyRunDescriptor {
    pub label: String,
    pub git_revision: Option<String>,
    pub similarity_algorithm: String,
    pub audio_algorithm: String,
    pub notes: Option<String>,
}

/// Row from `accuracy_run` for listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccuracyRunRow {
    pub id: i64,
    pub created_at: String,
    pub label: String,
    pub similarity_algorithm: String,
    pub audio_algorithm: String,
}

/// Open or create an accuracy database at `path` and apply schema.
pub struct AccuracyStore {
    conn: Connection,
}

impl AccuracyStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Insert one run and its per-fixture metrics; returns new `run_id`.
    pub fn record_run(
        &mut self,
        desc: &AccuracyRunDescriptor,
        fixtures: &[(String, DetectionAccuracyMetrics)],
    ) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        let created_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        tx.execute(
            "INSERT INTO accuracy_run (created_at, git_revision, label, similarity_algorithm, audio_algorithm, notes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                created_at,
                desc.git_revision,
                desc.label,
                desc.similarity_algorithm,
                desc.audio_algorithm,
                desc.notes,
            ],
        )?;
        let run_id = tx.last_insert_rowid();

        for (fixture_path, m) in fixtures {
            tx.execute(
                "INSERT INTO accuracy_fixture_result (
                   run_id, fixture_path, precision, recall, f1_score, timing_mean_abs_error_ms,
                   true_positives, false_positives, false_negatives, segments_detected
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    run_id,
                    fixture_path,
                    m.precision,
                    m.recall,
                    m.f1_score,
                    m.timing_mean_abs_error_ms,
                    m.true_positives as i64,
                    m.false_positives as i64,
                    m.false_negatives as i64,
                    m.segments_detected as i64,
                ],
            )?;
        }

        tx.commit()?;
        Ok(run_id)
    }

    /// Recent runs, newest first.
    pub fn list_runs(&self, limit: usize) -> Result<Vec<AccuracyRunRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, label, similarity_algorithm, audio_algorithm
             FROM accuracy_run ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(AccuracyRunRow {
                id: row.get(0)?,
                created_at: row.get(1)?,
                label: row.get(2)?,
                similarity_algorithm: row.get(3)?,
                audio_algorithm: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Load metrics for `fixture_path` from the last two runs that contain that fixture.
    pub fn last_two_metrics_for_fixture(
        &self,
        fixture_path: &str,
    ) -> Result<Option<(DetectionAccuracyMetrics, DetectionAccuracyMetrics)>> {
        let mut stmt = self.conn.prepare(
            "SELECT fr.precision, fr.recall, fr.f1_score, fr.timing_mean_abs_error_ms,
                    fr.true_positives, fr.false_positives, fr.false_negatives, fr.segments_detected
             FROM accuracy_fixture_result fr
             JOIN accuracy_run r ON r.id = fr.run_id
             WHERE fr.fixture_path = ?1
             ORDER BY fr.run_id DESC
             LIMIT 2",
        )?;

        let rows = stmt.query_map(params![fixture_path], |row| {
            Ok(DetectionAccuracyMetrics {
                precision: row.get(0)?,
                recall: row.get(1)?,
                f1_score: row.get(2)?,
                timing_mean_abs_error_ms: row.get(3)?,
                true_positives: row.get::<_, i64>(4)? as usize,
                false_positives: row.get::<_, i64>(5)? as usize,
                false_negatives: row.get::<_, i64>(6)? as usize,
                segments_detected: row.get::<_, i64>(7)? as usize,
            })
        })?;

        let mut buf = Vec::new();
        for r in rows {
            buf.push(r?);
        }
        if buf.len() < 2 {
            return Ok(None);
        }
        Ok(Some((buf[1].clone(), buf[0].clone())))
    }

    /// Latest metrics for a fixture, if any.
    pub fn latest_metrics_for_fixture(
        &self,
        fixture_path: &str,
    ) -> Result<Option<DetectionAccuracyMetrics>> {
        let mut stmt = self.conn.prepare(
            "SELECT fr.precision, fr.recall, fr.f1_score, fr.timing_mean_abs_error_ms,
                    fr.true_positives, fr.false_positives, fr.false_negatives, fr.segments_detected
             FROM accuracy_fixture_result fr
             WHERE fr.fixture_path = ?1
             ORDER BY fr.run_id DESC
             LIMIT 1",
        )?;

        Ok(stmt
            .query_row(params![fixture_path], |row| {
                Ok(DetectionAccuracyMetrics {
                    precision: row.get(0)?,
                    recall: row.get(1)?,
                    f1_score: row.get(2)?,
                    timing_mean_abs_error_ms: row.get(3)?,
                    true_positives: row.get::<_, i64>(4)? as usize,
                    false_positives: row.get::<_, i64>(5)? as usize,
                    false_negatives: row.get::<_, i64>(6)? as usize,
                    segments_detected: row.get::<_, i64>(7)? as usize,
                })
            })
            .optional()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracy::evaluate_detection_accuracy;
    use crate::segment_detector::{CommonSegment, MatchType};
    use tempfile::tempdir;

    fn sample_metrics() -> DetectionAccuracyMetrics {
        let detected = vec![CommonSegment {
            start_time: 0.0,
            end_time: 10.0,
            episode_list: vec!["a".into(), "b".into()],
            episode_timings: None,
            confidence: 1.0,
            video_confidence: None,
            audio_confidence: None,
            match_type: MatchType::Audio,
        }];
        let expected = vec![crate::accuracy::ExpectedFixtureSegment {
            start_time: 0.0,
            end_time: 10.0,
            min_episodes: 2,
        }];
        evaluate_detection_accuracy(&detected, &expected, false)
    }

    #[test]
    fn round_trip_run() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("acc.sqlite");
        let mut store = AccuracyStore::open(&db_path)?;

        let m = sample_metrics();
        let run_id = store.record_run(
            &AccuracyRunDescriptor {
                label: "unit".into(),
                git_revision: Some("abc".into()),
                similarity_algorithm: "Current".into(),
                audio_algorithm: "Chromaprint".into(),
                notes: None,
            },
            &[("synthetic/intro".into(), m.clone())],
        )?;
        assert!(run_id > 0);

        let listed = store.list_runs(5)?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].label, "unit");

        let latest = store
            .latest_metrics_for_fixture("synthetic/intro")?
            .expect("row");
        assert!((latest.f1_score - m.f1_score).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn last_two_for_fixture() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("acc.sqlite");
        let mut store = AccuracyStore::open(&db_path)?;

        let m1 = sample_metrics();
        store.record_run(
            &AccuracyRunDescriptor {
                label: "a".into(),
                git_revision: None,
                similarity_algorithm: "Current".into(),
                audio_algorithm: "Chromaprint".into(),
                notes: None,
            },
            &[("synthetic/intro".into(), m1.clone())],
        )?;

        let mut m2 = sample_metrics();
        m2.precision = 0.5;
        store.record_run(
            &AccuracyRunDescriptor {
                label: "b".into(),
                git_revision: None,
                similarity_algorithm: "Current".into(),
                audio_algorithm: "Chromaprint".into(),
                notes: None,
            },
            &[("synthetic/intro".into(), m2.clone())],
        )?;

        let pair = store
            .last_two_metrics_for_fixture("synthetic/intro")?
            .expect("two runs");
        assert!((pair.1.precision - 0.5).abs() < 1e-9);
        assert!((pair.0.f1_score - m1.f1_score).abs() < 1e-9);
        Ok(())
    }
}
