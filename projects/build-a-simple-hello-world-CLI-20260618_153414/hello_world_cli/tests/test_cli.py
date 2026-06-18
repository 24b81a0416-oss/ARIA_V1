<code>
import unittest
from unittest.mock import patch
from hello_world_cli import cli

class TestCLI(unittest.TestCase):
    @patch("sys.argv", ["cli.py", "--name", "ARIA"])
    def test_cli_with_name(self):
        with patch("builtins.print") as mock_print:
            cli.cli()
            mock_print.assert_called_once_with("Hello, ARIA!")

    @patch("sys.argv", ["cli.py"])
    def test_cli_without_name(self):
        with patch("builtins.print") as mock_print:
            cli.cli()
            mock_print.assert_called_once_with("Hello, World!")

if __name__ == "__main__":
    unittest.main()
</code>