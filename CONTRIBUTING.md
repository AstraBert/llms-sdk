# Contributing to `jake`

Thank you for your interest in contributing to this project! Please review these guidelines before getting started.

## Issue Reporting

### When to Report an Issue

- You've discovered bugs but lack the knowledge or time to fix them
- You have feature requests but cannot implement them yourself

> ⚠️ **Important:** Always search existing open and closed issues before submitting to avoid duplicates.

### How to Report an Issue

1. Open a new issue
2. Provide a clear, concise title that describes the problem or feature request
3. Include a detailed description of the issue or requested feature

## Code Contributions

### When to Contribute

- You've identified and fixed bugs
- You've optimized or improved existing code
- You've developed new features that would benefit the community

### How to Contribute

1. **Fork the repository and check out a secondary branch**

2. **Make your changes and test**

   ```bash
   jake build
   jake test
   # integration tests
   export OPENAI_API_KEY="..."
   export ANTHROPIC_API_KEY="..."
   export RUN_INTEGRATION_TESTS=true
   jake integration
   ```

   Ensure the build succeeds and all tests pass. Add tests for new features.

4. **Verify formatting and linting compliance**
   Ensure your changes pass all linting checks.

  ```bash
  jake clippy-fix
  jake format-fix
  ```

5. **Commit your changes**

6. **Submit a pull request**
   Include a comprehensive description of your changes.

---

**Thank you for contributing!**
