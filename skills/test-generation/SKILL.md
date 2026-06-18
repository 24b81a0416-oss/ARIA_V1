---
name: test-generation
description: Generate unit tests and integration tests for Python code
version: 1.0.0
---

# test-generation

## Purpose
Generate comprehensive test suites for Python modules. Cover normal cases, edge cases, and error cases.

## Instructions

When generating tests:

1. **Framework**: Use pytest by default, unittest if pytest is not available
2. **Coverage**: Test normal cases, edge cases (empty input, boundary values), and error cases (invalid input, exceptions)
3. **Fixtures**: Use pytest fixtures for reusable setup
4. **Mocking**: Use unittest.mock for external dependencies
5. **Naming**: `test_<function>_<scenario>` pattern

Output format:
- One test file per source module: `test_<module>.py`
- Include setup instructions if additional packages are needed
- Mark tests that require mocking with a comment

Be thorough. Aim for >80% coverage on core logic.
