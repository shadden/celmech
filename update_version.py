#!/usr/bin/python
# This script automatically creates a list of examples by reading the header in all problem.c files.
import glob
import subprocess
ghash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

with open("version.txt") as f:
    celmechversion = f.readlines()[0].strip()
    print("Updating version to "+celmechversion)

with open("src/disturbing_function.c") as f:
    celmechlines = f.readlines()
    for i,l in enumerate(celmechlines):
        if "**VERSIONLINE**" in l:
            celmechlines[i] = "const char* celmech_version_str = \""+celmechversion+"\";         // **VERSIONLINE** This line gets updated automatically. Do not edit manually.\n"

    with open("src/disturbing_function.c", "w") as f:
        f.writelines(celmechlines)

with open("setup.py") as f:
    setuplines = f.readlines()
    for i,l in enumerate(setuplines):
        if "version='" in l:
            setuplines[i] = "    version='"+celmechversion+"',\n"
        if "GITHASHAUTOUPDATE" in l:
            setuplines[i] = "    ghash_arg = \"-DCELMECHGITHASH="+ghash+"\" #GITHASHAUTOUPDATE\n"

    with open("setup.py", "w") as f:
        f.writelines(setuplines)

# find changelog
with open("changelog.md") as f:
    found_start = 0
    changelog = ""
    cl = f.readlines()
    for l in cl:
        if found_start == 0 and l.startswith("### Version"):
            if celmechversion in l:
                found_start = 1
                continue
        if found_start == 1 and l.startswith("### Version"):
            found_start = 2
        if found_start == 1:
            changelog += l

if found_start != 2 or len(changelog.strip())<5:
    raise RuntimeError("Changelog not found")

with open("_changelog.tmp", "w") as f:
    f.writelines(changelog.strip()+"\n")

print("----")
print("Changelog:\n")
print(changelog.strip())
print("----")
print("Next:")
print("\ngit commit -a -m \""+celmechversion+"\"")
print("git tag "+celmechversion+" && git push --tags")
print("gh release create "+celmechversion+" --notes-file _changelog.tmp")
