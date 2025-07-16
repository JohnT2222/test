# check_requirements.py

import pkg_resources

with open("requirements.txt") as f:
    required = pkg_resources.parse_requirements(f.readlines())

installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

print("\n🔍 Checking installed requirements:\n")
for req in required:
    name = req.name.lower()
    required_version = str(req.specifier)

    if name in installed_packages:
        installed_version = installed_packages[name]
        if req.specifier.contains(installed_version, prereleases=True):
            print(f"✅ {name}=={installed_version} (matches requirement {required_version})")
        else:
            print(f"⚠️ {name}=={installed_version} (does NOT match requirement {required_version})")
    else:
        print(f"❌ {name} is NOT installed")
