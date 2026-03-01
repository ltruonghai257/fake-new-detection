---
trigger: model_decision
---

# Expanded Coding Guidelines

## Core Principles

-   My project's programming language is python
-   Use early returns when possible
-   Always add documentation when creating new functions and classes

## Expanded Rules

### 1. Early Returns - Comprehensive Rules

-   Return immediately upon detecting invalid input or error conditions
-   Avoid deeply nested if-else statements by returning early from error paths
-   Place validation and guard clauses at the beginning of functions
-   Use early returns for both success and failure paths when appropriate
-   Return None or raise exceptions early rather than continuing with invalid data
-   Early returns improve readability by reducing indentation levels

### 2. Documentation Requirements

#### For All New Functions

-   Write docstrings for every function (no exceptions)
-   Include a one-line summary of what the function does
-   Document all parameters with their expected types and purposes
-   Document the return value and its type
-   Document any exceptions that might be raised
-   Include at least one usage example in docstring
-   Use clear, concise language that explains the "why" not just the "what"

#### For All New Classes

-   Write docstrings for every class definition
-   Include a one-line summary of the class purpose
-   Document all class attributes and their types
-   Document the `__init__` method's parameters
-   Document any important methods with their own docstrings
-   Include usage example showing how to instantiate and use the class
-   Explain the relationship between class methods when relevant

#### Documentation Format

-   Use triple-quoted strings (""") for all docstrings
-   Follow consistent formatting across all documentation
-   Keep examples simple and runnable
-   Explain any non-obvious parameters or behaviors

### 3. Python-Specific Rules

-   Use type hints on function parameters and return types
-   Write code that follows PEP 8 style conventions
-   Use meaningful variable names (avoid single letters except in loops)
-   Import statements should be organized: standard library, third-party, local
-   Use f-strings for string formatting instead of .format() or %
-   Prefer list comprehensions for simple iterations
-   Use context managers (with statements) for file and resource handling

### 4. Function Design Rules

-   Functions should do one thing well (single responsibility)
-   Keep function length reasonable (aim for under 20 lines when possible)
-   Function names should be descriptive action verbs (e.g., `calculate_`, `validate_`, `process_`)
-   Avoid mutable default arguments
-   Use early returns to exit when preconditions aren't met
-   Place all validation logic at the function's beginning

### 5. Class Design Rules

-   Class names should be nouns (e.g., `DataProcessor`, `ModelEvaluator`)
-   Keep classes focused on a single responsibility
-   Document the purpose and usage of each class clearly
-   Initialize all instance attributes in `__init__`
-   Use private methods (prefix with `_`) for internal helper methods
-   Consider using dataclasses for simple data-holding classes

### 6. Error Handling Rules

-   Always validate input parameters at function start and return early if invalid
-   Use specific exception types (ValueError, TypeError, etc.) rather than generic Exception
-   Document which exceptions a function can raise in its docstring
-   Use early returns to avoid error-handling nesting
-   Provide helpful error messages that explain what went wrong

### 7. Code Readability Rules

-   Use early returns to keep function complexity low
-   Avoid deeply nested code (max 3 levels of nesting)
-   Add comments only for "why" decisions, not "what" the code does
-   Use whitespace and blank lines to group related logic
-   Break long lines at logical points (max 88 characters recommended)
-   Use descriptive variable names that explain intent

### 8. Testing Rules

-   Write docstrings with example usage that could serve as basic tests
-   Create separate test files for complex functions
-   Document any test coverage expectations in class/function docstrings
-   Examples in docstrings should be valid Python that could run
