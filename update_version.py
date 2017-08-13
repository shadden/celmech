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

print("To commit, copy and paste:")
print("\ngit commit -a -m \"Updating version to "+celmechversion+"\"")
