# JUST TAKE IT DOWN

- [JUST TAKE IT DOWN](#just-take-it-down)
  - [Preface](#preface)
  - [Vim](#vim)
    - [Scrolling](#scrolling)
    - [Motion](#motion)
      - [Common Motion](#common-motion)
      - [Jump Motion](#jump-motion)
    - [Change Text](#change-text)
    - [Buffer, Window and Tab in VIM](#buffer-window-and-tab-in-vim)
      - [Window Commands](#window-commands)
      - [Terminal Window In VIM](#terminal-window-in-vim)
      - [Tab Page Commands and Operations](#tab-page-commands-and-operations)
    - [Editing](#editing)
  - [Markdown(CommonMark)](#markdowncommonmark)
    - [Paragraph](#paragraph)
    - [Links and Images](#links-and-images)
  - [In Case I Forget Again](#in-case-i-forget-again)

## Preface

OK, I have to admit, I am not a fan of taking notes whenever I am learning something new.\
You know, you will just tell yourself "Come on, of course I can remember this" and invariably turn out\
to forget every detail you should pay attention to.

Although I have this disagreement against `好记性不如烂笔头` as I reckon that the capability of memorizing\
is actually a result of training and the more we rely on notebooks just to remember things, the worse our \
memory become, and we need to **construct** a system, not merely "remember" all these pieces of fragments\
I do have perceived the necessity to just take things down and fathom deeper during this process.

## Vim

### Scrolling

|Key|Descrition|
--|--
|**Ctrl + D**|Scroll `'scroll'` lines down|
|**Ctrl + U**|Scroll `'scroll'` lines up|
|**Shift + Up**|PageUp|
|**Shift + Down**|PageDown|
|{count} **Ctrl + E**|Scroll {count} lines down|
|{count} **Ctrl + Y**|Scroll {count} lines up|

_In a common keyboard, **E is near D** and **Y is near U**._

### Motion

#### Common Motion

|Operation|Description|
|--|--|
|**g0**, **g[home]**|To the first character of current **screen line**|
|**g^**|To the first __non-blank__ character of current **screen line**|
|**gm**|To the middle of the current **screen line**|
|**g$**|To the last character of current **screen line**|
|{count} **\|**|To the {count}-th column of the current line|
|**f**{char}|To the [count]-th occurrence of {char} to the **left**<br>(the cursor is placed **on** {char}).|
|**F**{char}|Like f{char}, but to the right of the cursor|
|**t**{char}|To the [count]'th occurrence of {char} to the **left**<br>(the cursor is placed **before** {char})|
|**T**{char}|Like t{char}, but to the right of the cursor|
|**;**|Repeat the latest f, F ,t or T|
|{count} **-**|{count} lines upwards, on the **first non-blank** character|
|**Enter, +**|{count} lines downwards, on the **first non-blank** character|
|{count}**e**|Forward to the **end** of the word {count} times|
|{count}**ge**|Backward to the **end** of the word {count} times|
|{count}**aw, as, ap, a(** _etc._|Select a word(include the blank before the word)<br>,a sentence,a paragraph, a () block, etc.|
|{count}**iw, is, ip, i(** _etc._|Select a inner word(just the word), a inner sentence,<br>a inner paragraph or inside a () block, etc.|

#### Jump Motion

1. General Jump

   |Key|Description|
   |--|--|
   |{count}`C-o`|To the {count} older cursor position.|
   |{count}`C-i`|To the {count} newer cursor position.|

   Use `:ju(mps)` to check all the jumps, and `cle(arjumps)` to clear the jump list.

2. Change List Jump

   |Operation|Description|
   |--|--|
   |{count}**g;**|To the {count} older position in change list.|
   |{count}**g,**|To the {count} newer position in change list.|

   Use `:changes` to check all the changes made in current window.

### Change Text

|Operation|Description|
|--|--|
|{count}**c**{motion}|Delete {motion} text and start insert.|
|{count}**D**|Delete until the end of the current line and {count}-1 more lines.|
|{count}**C**|Delete until the end of the current line and {count}-1 more lines, then enter Insert Mode.|
|{count}**s**|Delete {count} characters and start insert.|
|{count}**S**|Delete {count} lines and start insert.|
|{count}**J**|Join {count} lines, remove the indent and insert up to two spaces.|
|{count}**gJ**|Like **J** but do not delete any space.|
|{count}**~**|Switch case of {count} characters from the cursor.|
|**gu**{motion}|Make {motion} text lowercase.|
|**gU**{motion}|Make {motion} text uppercase.|
|{count}**guu**|Make {count} lines lowercase from the current line|
|{count}**gUU**|Make {count} lines uppercase from the current line|

More at `help change.txt`.

### Buffer, Window and Tab in VIM

> A buffer is the in-memory text of a file.\
> A window is a viewport on a buffer.\
> A tab page is a collection of windows.

#### Window Commands

1. Open and Close
   1. Common open and close command:

      | Command|Description|Shortcut Key|
      |:--|:--|--|
      |**sp(lit)** {file}|if file is given, split the window and edit the file<br>otherwise vim\'ll open the current buffer.|`c-w,s`|
      |**vs(plit)**|like `:split` but split vertically.|`c-w,v`|
      |**new** |create a new window to edit a new file.|`c-w,n`|
      |**vne(w)**|like :new but split vertically.||
      |**vert(ical)** {cmd}|Execute {cmd}, but if the command<br>splits a window, it splits vertically.<br>For example, `:vert help window`.||
      |{count}**q(uit)**|Quit the {count} window if {count} is given,<br>otherwise quit the current one.|`c-w,q`|
      |{count}**clo(se)**|Like `:quit` but will fail when there is only one window.|`c-w,c`|
      |{count}**on(ly)**|Leave only the current window or {count} windows open<br>and close all the rest.|`c-w,o`|

      |Key|Description|
      --|--
      |**Ctrl-W + H/J/K/L**|Move the **current window** to the given direction.|
      |**Ctrl-W + h/j/k/l**|Move the **cursor** to the window in the given direction.|

   2. Involving **tag** under the cursor

      There command rely on **tags file** and make sure you have generated one under the working directory with `ctags -R`.

      |Command|Description|Shortcut Key|
      |:--|:--|--|
      |**sta(g)** [tagname]|Create a window and jump to the tag<br>under the cursor(use shortcut key)<br>or the tag given(use command).|`c-w, ]`|
      |**pta(g)** [tagname]|Create a _preview window_ and jump to the tag<br>under the cursor(use shortcut key)<br>or the tag given(use command).|`c-w, }`|

      More at `help window`-`Tag or file name under the cursor` and `The preview window`
      
   3. Involving **include search**

      |Shortcut Key|Description|
      |--|--|
      |`C-W,i`|Open a new window, with the cursor on the first line<br>that contains the keyword under the cursor|

	  More at `help tagsrch`-`6. Include File Search`

#### Terminal Window In VIM

Basic use:
- Use `:ter(minal) {command}` to start a terminal emulator and execute {command}, and\
  the outputs of the {command} will be showed on a new window. If the {command} is not given\
  VIM will open a new shell window.

Terminal-Job and Terminal-Normal mod
- In Terminal-Job mod, all the typed keys are sent to the terminal\
  while in Terminal-Normal mod, the contents of the terminal become common string text\
  of a un-modifiable buffer so you can use usual VIM command.
- Use `C-W,n` in Terminal-Job mod to switch to Terminal-Normal mode.\
  And any action that would start insert mode(`i`,`a`,etc.) under Terminal-Normal mode\
  will invoke a switch to Terminal-Job mod.
- In Terminal-Job mode, you can also use `C-W,:` to enter vim command line.
  

More at `help terminal`

#### Tab Page Commands and Operations

|Command|Description|
|--|--|
|{count}**tabe(dit)**, **tabnew** {file}|Open a new tab page after the current tab page.|
|{count}**tab** {cmd}|Execute the {cmd},but when it create a new window, create a new tab instead.|
|{count}**tabc(lose)(!)**|Close tab page {count},<br>`:tabc 2`:Close the second tag page<br>`:-2tabc`:Close the second previous tab page<br>`:tabc $`:Close the last tab page|
|{count}**tabo(nly)**|Close all tab pages except {count} one|
|**tabm(ove) +[N]/-[N]**|Move the current tab page [N] places to the right(+) or left(-)|

|Operation|Description|Shortcut Key|
|--|--|--|
|**gt/gT**|Jump to the next/previous tab page|`C-PgDn`/`C-PgUp`|


### Editing

|Command|Description|
|--|--|
|**e(dit)(!)** {file}|Edit or re-edit(when the current file has been modified outside.Add `!`<br>will discard all the changes made to the current buffer) the file.|
|**fin(d)** {file}|Find {file} in `path` and edit it.|

## Markdown(CommonMark)

### Paragraph

> A paragraph is [consecutive](https://www.merriam-webster.com/dictionary/consecutive "means following one after the other in order") lines of text with **one or more blank lines between them.**
>
> For a **line break**, add either a **backslash \\** or **two blank spaces** at the **end of the line**.\
> (Also you can use HTML label `<br>`)

### Links and Images

> Link text is enclosed by **square brackets []**, and for inline links, the link URL is enclosed by **parens ()**. 

	[text](http://a.com "title")
	
	[text][id]
	⋮
	[id]: http://b.org/ "title"

Images are almost identical to links, but a image starts with `!`.

	![alternative text](http://a.com/b.jpg "title")
	
	![alternative text][id]
	⋮
	[id]: http://b.org/c.jpg "title"

_More at [CommonMark Spec](https://spec.commonmark.org) and [CommonMark Tutorial](https://commonmark.org/help/tutorial/)_

_GFM(Github Favored Markdown) is known as a strict superset of the CommonMark, see its extensions to CommonMark [here](https://github.github.com/gfm/)_

_See also: [Google Markdown Style Guide](https://github.com/google/styleguide/blob/gh-pages/docguide/style.md)_ 

## In Case I Forget Again

**READING!!** not just and tech books and textbooks, **EXPAND** yourself!!

- [ ] 《呼吸》特德姜
- [ ] 《新名字的故事》伊莲娜
- [ ] 《刀锋》毛姆

and

- [ ] 《西部世界》03.15
- [ ] 《环形物语》04.03
