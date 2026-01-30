<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# setup

## Purpose
Interactive configuration wizard that guides users through MouseReach system setup, handling platform-specific environment variable configuration (Windows setx vs Unix .bashrc), path validation, and PowerShell execution policy fixes for network drive conda environments. Configures NAS drive location and pipeline processing root paths.

## Key Files
| File | Description |
|------|-------------|
| `wizard.py` | Interactive CLI setup wizard - prompts for NAS drive and processing root paths, validates, saves environment variables persistently (Windows setx / Unix .bashrc) |
| `fix_powershell.py` | PowerShell execution policy fixer - sets execution policy to Bypass for CurrentUser to allow conda activation from network drives (Y:, etc.) |
| `__init__.py` | Module marker |

## For AI Agents

### Working In This Directory
- The wizard configures paths: `MouseReach_PROCESSING_ROOT` (required) and `MouseReach_NAS_DRIVE` (optional)
- No hardcoded defaults - users must run `mousereach-setup` to configure their paths
- Platform detection: Windows uses `setx` for persistent env vars (effective in new sessions), Unix appends to `~/.bashrc`
- Path validation: Wizard can create directories if they don't exist, with user confirmation
- PowerShell fix only needed on Windows when using conda environments stored on network drives (UNC paths or mapped drives)

### CLI Commands
```bash
mousereach-setup              # Run interactive configuration wizard
mousereach-setup --show       # Display current configuration
mousereach-setup --help       # Show help message
mousereach-fix-powershell     # Fix PowerShell execution policy for network drives
```

## Dependencies

### Internal
None - standalone utility module

### External
- `os`, `sys` - Environment variable and platform detection
- `subprocess` - Running setx (Windows) and PowerShell commands
- `pathlib` - Path validation and manipulation

<!-- MANUAL: -->
