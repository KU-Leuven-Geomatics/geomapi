---
marp: false
paginate : true
headingDivider: 4
---

# Packaging
Everything you need to know to write and share your Python code



## Structure
In order to make it easier to reuse your code, you can put it in a package. However, there are certain standards your package should conform to.

### Project Structure
Here you can find the basic structure to organize your project. [more info](https://docs.python-guide.org/writing/structure/)

```yaml
"project"/            # The root of the repository
│
├─ venv/              # Your virtual environment
├─ "package"/         # The name of your package
│  ├─ __init__.py     # The init file to show python this folder is a package
│  ├─ "module".py     # modules ending in .py
│  ├─ "subpackage"/   # sub-packages
├─ docs/              # Your documentation for the package
├─ test/              # The unit test of your code
├─ README.md          # A file with the basic information about the project
├─ requirements.txt   # The dependencies of your project during development
├─ .gitignore         # A file to indicate which files to ignore
├─ LICENSE            # The (open-source) license
```


### Package Structure
A package is a folder containing an `__init__.py` file and a collection of modules & subpackages.
```yaml
"mypackage"/         # The main package
├─ __init__.py        # The init file to show python this folder is a package
├─ "module1".py        # modules ending in .py
├─ "module2".py
├─ ...
├─ "subpackage"/       # sub-package
│  ├─ __init__.py      # The init file to show python this folder is a package
│  ├─ "submodule".py   # modules ending in .py
│  ├─ ...
```

#### Importing in the package
Your code is designed to run outside of the package. Since python is very strict about where the code is run from, you will need to import your relative modules as if you are importing them from outside of the package ex:
```py
# to import module1.py in module2.py
import mypackage.module1
```

## Testing
To remove the guesswork in evaluating if your package works properly, we use [unittests](https://docs.python.org/3/library/unittest.html). These are a collection of functions that test the base functionalities of all the functions and classes in the package.

### Test file
Each `module.py` should have a corresponding `test_module.py` in the `tests/` folder. To test the functionality, call every function and compare the output with what you expect.
```py
import unittest

class TestMyClass(unittest.TestCase):

    def test_function_1(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_function_2(self):
        self.assertEqual('FOO'.lower(), 'foo')


if __name__ == '__main__':
    unittest.main()
```
### Test Running
To run all the tests simply run the following command:
```bash
python -m unittest
```

## Documentation
Keeping your code readable with [docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [sphinx](https://www.sphinx-doc.org/en/master/)

### Source code documenting
If you want others (or yourself in the future) to understand your code, you should explain the functionality of all your functions and classes. Use [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) to automatically generate the template.
```py
def function(value : str) -> str:
    """_summary_

    Args:
        value (str): _description_

    Returns:
        str: _description_
    """
    return value
```

### Documentation generation
Use [Sphinx](https://www.sphinx-doc.org/en/master/) to generate automatic documentation of the full API. You can create websites using some pre-made templates to easily organize your documentation.

#### Setup
[Sphinx](https://www.sphinx-doc.org/en/master/) provides a handy `quickstart` function to create the source files needed to generate the documentation

``` sh
cd ./docs                         # Navigate to the docs folder
sphinx-quickstart                 # set up the docs sources
```

#### Autodoc
[Sphinx](https://www.sphinx-doc.org/en/master/) can automatically create the documentation structure, based on the package and its modules. Activate these extensions in `conf.py`

``` py
extensions = [
    'sphinx.ext.autodoc',           # search for docstrings
    'sphinx.ext.autosummary',       # Search for summaries
    'sphinx.ext.napoleon',          # Allow for napoleon style docstrings
]
autosummary_generate = True         # Turn on sphinx.ext.autosummary
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
```
Generate the source framework using APIdoc
```
sphinx-apidoc -o . ../mypackage/ -e
```

#### Build the documentation
The HTML code can be easily generated using the make file in the `docs/` directory  
```
./make html
```

### Documentation hosting

#### Gitlab
gitlab.kuleuven.be has a webserver to host your documentation per project by the following structure points to the `public/` folder in the root of the repo:
https://"group".pages.gitlab.kuleuven.be/"collection"/"project"/

for example: https://geomatics.pages.gitlab.kuleuven.be/research-projects/geomapi

#####  CI/CD File
Create `.gitlab-ci.yml` in the root of the project with the following content to automatically publish the page after each push using [CI/CD](https://docs.gitlab.com/ee/ci/):
```yaml
pages:
  stage: deploy
  script:
    - mkdir .public
    - cp -r * .public
    - mv .public docs
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

#### Github
Github works with the github actions automation system. a yaml based instruction platform. You can enable the github pages in your project settings by selecting the `gh-pages` branch after the workflow has run at least once.

##### workflows File
Create a `.github/workflows/'name'.yml` file and 
```yaml
name: Docs
on: [push, pull_request, workflow_dispatch]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install -r docs/docs_requirements.yaml
      - name: Sphinx build
        run: |
          sphinx-build docs/source _build
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
        with:
          publish_branch: gh-pages
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: _build/
          force_orphan: true
```

## Publishing
To publish your package to pypi.org and Conda you'll need `setuptools`


### Required Files
Files that are nescacary to build and deploy your package.

#### pyproject.toml

``` 
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

#### setup.cfg
```
[metadata]
name ="name"
version = 0.0.1
author = "author"
description = "desc"
long_description = file: README.md
long_description_content_type = text/markdown
url ="url"
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: GNU General Public License (GPL)
    Operating System :: OS Independent

[options]
packages = find:
python_requires = >=3.8
install_requires =
      "dependancies"
```

#### .pypirc
a file in your $HOME directory to store the publish token

```
[distutils]
index-servers =
    pypi
    testpypi
    
[pypi]
  username = __token__
  password = "***"

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = "***"
```


### Build commands
It is advised to first publish to the testpypi server before the real one.
Create an account and an token on the websites.

```
python pip -m build
python3 -m twine upload --repository testpypi dist/*
python3 -m twine upload dist/*
``` 


## Automation

### Gitlab
[CI/CD](https://docs.gitlab.com/ee/ci/) or Continuous integration and deployment can automate most of these tasks
by using a `.gitlab-ci.yml`file. This performs a tasks in multiple stages:

#### Stage Preparation
Set up a virtual environment and instal the dependencies to validate the documentation and package
``` yamp
variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
cache:
  paths:
    - .cache/pip
    - venv/
stages:
  - pages
  - build
  - deploy
before_script:
  - python -V
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  #- pip install -r requirements.txt
```

#### Pages Stage
We use  `Moc-imports` in the `conf.py` to import all the dependencies so we don't have to import them all again.
``` yaml
pages:
  stage: pages
  script:
    - pip install sphinx 
    - pip install sphinx_rtd_theme
    - sphinx-build -b html docs/source/ public/
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

#### Build Stage
Next we build the package using `setup.cfg`.
``` yaml
build:
  stage: build
  artifacts:
    paths:
      - build
      - dist
  script:
    - pip install --upgrade build
    - python -m build
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

```

#### Deploy Stage
The final step is publishing the package to a CDN, in this case pypi. Use the token from pypi in the environment variables in Gitlab to hide the password.
``` yaml
deploy:
  stage: deploy
  script:
    - pip install --upgrade build twine
    - python -m build
    - TWINE_PASSWORD=${PYPI_TOKEN} TWINE_USERNAME=__token__ python -m twine upload --verbose dist/*
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  allow_failure: true
``` 

### Github

You can read the [Docs](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) for more info and starter templates.