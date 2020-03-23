# JUST TAKE IT DOWN

## VIM

### Some Special Keys in *Insert Mode*

1. Editing

   |Key|Description|
   ---|---
   |**Ctrl + R**+[Reg]|Paste text in register [Reg]|
   |**Ctrl + U**|Delete the content before the cursor in the current line.|
   |**Ctrl + W**|Delete the front part of the word under the cursor.|

2. Scrolling

   |Key|Description|
   --|--
   |**Shift + Up**|PageUp|
   |**Shift + Down**|PageDown|
   |Ctrl-X Ctrl-E|Scroll one line down|
   |Ctrl-X Ctrl-Y|Scroll one line up|
   
   Note: Ctrl + X actually trigger a sub-mode in insert mode, in this mode each Ctrl+E(Ctrl+Y) scrolls window one line down(up).

3. Motion

   |Key|Description|
   --|--
   |**Shift + Left**|Move a word back|
   |**Shift + Right**|Move ward forward|
   
4. Completion(In **Ctrl-X Mode**)

   |Key|Description|
   --|--
   |Ctrl-X **Ctrl-L**|Complete a whole line.|
   |Ctrl-X **Ctrl-K**|Complete a word with the `'dictionary'` file.|
   |Ctrl-X **Ctrl-T**|Complete a word with the `'thesaurus'` file.|
   
   Note: You can use `Ctrl + N`/`Ctrl + P` to apply next/previous completion.

   More at `help insert`-`Insert Mode Completion`.



