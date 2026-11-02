Decision 3: How does the system recognize a document?

Option 1: by file name. The document is known by where it lives, like tutorial/query-params.md.

Easy to read.
If you edit the file, the system doesn’t notice. If you rename it, the system thinks it’s a new document.

Option 2: by content. The system makes a short fingerprint from what’s inside the file. Same content always gives the same fingerprint.

Loading the same file twice never creates a duplicate.
If the file is edited, the system notices because the fingerprint changes.
The file name is still saved alongside it so you can read it.

My pick: Option 2. It’s safer, and it’s an easy thing to explain in an interview.

Decision 4: What happens when a file fails to load?

Say you load 200 files and file #57 is corrupt.

Option 1: stop everything. The loader crashes and shows the error.

You notice problems right away.
One bad file blocks the other 199.

Option 2: skip it and keep going. The bad file is skipped, the reason is written to a log, and a summary is printed at the end (for example: 198 loaded, 2 skipped, with reasons).

One bad file doesn’t block the rest.
Nothing is hidden, because the log and summary show exactly what was skipped.

My pick: Option 2. Real document collections always have a few bad files, and the skipped list is useful material for your “what did not work” section later.

One last thing is open: the documents you’ll use.

FastAPI docs (my pick): technical, full of exact names and settings, which is where keyword search shows its value.
GitLab handbook: feels more like real company docs, but has fewer exact terms.
