# Security Policy

## Supported versions

Only the latest `main` branch is supported. Tagged releases are provided as
historic reference points and will not receive backported fixes.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Instead:

1. Open a [private security advisory](../../security/advisories/new) on
   GitHub, or
2. Send an email to the maintainer listed in `pyproject.toml`.

You should receive an acknowledgement within a few days. Fixes are
coordinated with the reporter before a public disclosure.

## Scope

This repository is a research codebase. It:

- does **not** process untrusted data in a production setting,
- does **not** ship trained model weights (see `data/README.md` for provenance
  of any weights you train yourself),
- does **not** expose a network service.

Classical web-app vulnerability classes therefore do not apply. Reports about
supply-chain risks, unsafe deserialisation (`torch.load` on untrusted files),
or credential leaks are welcome.
