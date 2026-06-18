---
name: code-review
description: Review Python code for bugs, security issues, performance problems, and style violations
version: 1.0.0
---

# code-review

## Purpose
Review code thoroughly and provide actionable feedback. Focus on correctness, security, and maintainability.

## Instructions

When reviewing code, check for these issues in order:

1. **Correctness**: Logic errors, off-by-one, edge cases, race conditions
2. **Security**: Injection, sensitive data exposure, auth bypasses
3. **Performance**: N+1 queries, unnecessary allocations, O(n²) algorithms
4. **Maintainability**: Dead code, complex conditionals, missing abstractions
5. **Style**: Inconsistent naming, missing type hints, formatting

Output format:
- ✅ PASS for clean files
- ⚠️ ISSUE for each problem found (file:line - description)
- 📊 Summary with issues by severity

Be specific. Include file paths and line numbers when possible.
