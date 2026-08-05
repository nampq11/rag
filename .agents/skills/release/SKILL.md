---
name: release
description: Creates a versioned GitHub release from the repository default branch. Use when asked to "create a release", "cut a release", "publish a release", "release a new version", or "tag a release". Updates versions and changelog entries, validates the release, pushes an annotated tag, and creates the GitHub release.
---

Create a release only from an up-to-date default branch.

## Step 1: Inspect the release state

1. Read the `commit` and `update-changelog` skills before making a release commit or changing the changelog.
2. Run `pwd`, `git branch --show-current`, `git status --short`, and `git remote -v`.
3. Identify the default branch with `gh repo view --json defaultBranchRef`.
4. Fetch the remote, switch to the default branch, and fast-forward it with `git pull --ff-only origin <default-branch>`.
5. Abort if the working tree contains unrelated changes, the branch cannot fast-forward, or the latest tag already matches the requested version.

## Step 2: Choose the version

Use the version the user supplies. If none is supplied, inspect the changes since the latest tag and ask the user to choose the semantic-version bump.

| Change type | Bump |
|---|---|
| Breaking API or behavior | Major |
| Backward-compatible feature | Minor |
| Backward-compatible bug fix | Patch |

Find the baseline and changes with:

```bash
git describe --tags --abbrev=0
git log <latest-tag>..HEAD --oneline
```

## Step 3: Prepare the release commit

1. Update `CHANGELOG.md` using the changelog skill. Move the relevant `Unreleased` entries into a dated heading for the new version, leaving an empty `Unreleased` heading above it.
2. Update the project version with `uv version <version>` or `uv version --bump <major|minor|patch>`.
3. Run `uv lock` if the version command did not update `uv.lock`.
4. Review the diff. Ensure the changelog, project metadata, and lockfile all report the same version.
5. Run the repository's formatter, linter, and test suite.
6. Commit only the release files using the repository's commit convention, then push the default branch.

## Step 4: Publish the release

1. Verify the default branch is clean and contains the release commit.
2. Create an annotated tag named `v<version>` and push it:

```bash
git tag -a v<version> -m "v<version>"
git push origin v<version>
```

3. Create a non-draft, non-prerelease GitHub release with `gh release create`. Use a concise body derived from the released changelog entries.
4. Verify it with:

```bash
gh release view v<version> --json url,tagName,targetCommitish,isDraft,isPrerelease
```

## Completion criteria

Report the release URL, version, release commit, and validation commands. Do not create the tag or GitHub release when the release commit has not been pushed successfully.
