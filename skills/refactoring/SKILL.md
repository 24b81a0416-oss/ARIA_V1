---
name: refactoring
description: Analyze and refactor Python code to improve structure, readability, and maintainability
version: 1.0.0
---

# refactoring

## Purpose
Analyze code structure and suggest concrete refactoring improvements. Always preserve behavior.

## Instructions

When refactoring code:

1. **Analyze first**: Identify code smells (long functions, deep nesting, duplicated code, large classes)
2. **Prioritize**: Focus on changes with the highest impact-to-risk ratio
3. **Patterns**: Suggest specific refactoring patterns:
   - Extract method/function
   - Replace conditional with polymorphism
   - Introduce parameter object
   - Decompose conditional
   - Replace temp with query
4. **Preserve behavior**: Never change the public API without explicit approval
5. **Show diffs**: Provide before/after for each change

Output format:
- 🔍 Analysis of code smells found
- 📋 List of suggested refactorings with priority (high/medium/low)
- 📝 For each: before code → after code with explanation
- ⚠️ Risk assessment for each change
