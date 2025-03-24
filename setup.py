from setuptools import setup, find_packages
import os
import re

def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "sqlbear", "_version.py")
    with open(version_file, "r") as f:
        content = f.read()
    
    # Use regex to extract the version correctly
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)

    raise RuntimeError("Version not found in sqlbear/_version.py")



setup(
   name="sqlbear",
   version=get_version(),
   packages=find_packages(),
   install_requires=["pandas", "sqlalchemy", "pytz"],
   author="Your Name",
   author_email="your.email@example.com",
   description="A pandas extension for easy SQL interaction",
   long_description=open("README.rst", encoding="utf-8").read(),
   long_description_content_type="text/x-rst",  # Specify .rst format
   url="https://github.com/readytowin2298/sqlbear",
   classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent",
   ],
   python_requires=">=3.7",
)