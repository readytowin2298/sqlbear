from setuptools import setup, find_packages


try:
    from sqlbear._version import __version__
except ImportError:
    __version__ = "unknown"


setup(
   name="sqlbear",
   version=__version__,
   packages=find_packages(),
   install_requires=["pandas", "sqlalchemy", "pytz"],
   author="Your Name",
   author_email="your.email@example.com",
   description="A pandas extension for easy SQL interaction",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   url="https://github.com/readytowin2298/sqlbear",
   classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent",
   ],
   python_requires=">=3.7",
)