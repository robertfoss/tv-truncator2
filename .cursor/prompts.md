# Testing Enforcement Prompts

## Before Making Changes
"Before implementing this feature, write comprehensive tests that cover:
1. Happy path scenarios
2. Edge cases and error conditions  
3. Input validation
4. Expected outputs

Run `cargo test` to ensure all existing tests still pass."

## After Making Changes
"Now run `cargo test` to verify:
1. All new tests pass
2. No existing tests were broken
3. Code coverage is maintained
4. Performance benchmarks are met

If any tests fail, fix the issues before proceeding."

## Code Review Checklist
- [ ] Tests written before implementation (TDD)
- [ ] All tests pass (`cargo test`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Code formatted (`cargo fmt`)
- [ ] Documentation updated
- [ ] Error handling comprehensive
