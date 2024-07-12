---
marp: true
paginate : true
headingDivider: 4
---
# Testcase Creation

The DO's and DON'T's of creating properly documented testcases

## Setup

Install pytest

```py
pip install -U pytest
```

## File structure

Use one main `.ipynb` file, located at `docs/source/testcases/` for you specific testcase.

## Resource location

To reference images in the notebook, they should be placed in `/docs/pics/`. They can than easily be referenced using `!["alt text"](../../pics/"imagename")`
!["alt text"](../../pics/NewGeometry.PNG)

## Images

You can layout images using:

```html
<img src="/imagepath" width = "49%">
```

by altring the width, you can put multiple images side by side

## Equations

Latex based dollar math is supported like this:

```md
$$ progress= 1 \text{  if  } PoC \geq 0.3 \\ 
0 \text{  else  } $$
```

$$
progress= 1 \text{  if  } PoC \geq 0.3 \\ 
0 \text{  else  }

$$

## Notebook excecution

By default, to prevent building errors, the notebooks are not executed during documentation building. Therefor, you should save the output in seperate markdown blocks.

## Content

The noteboook should be structured as if you are starting from scratch and going over the full process, explaining the thinking behind each function and illustrate properly with images if possible.
