"""PyInstaller entry point for the self-contained desktop distribution."""
from grand_council.desktop import main

if __name__ == '__main__':
    raise SystemExit(main())
