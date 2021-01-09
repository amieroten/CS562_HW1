''' A short script that takes as an argument a filename
    pointing to an .xml.gz file and deserializes, then
    parses the xml contained, spitting out all paragraphs
    <P> that are part of the <TEXT> section of all <DOC>
    sections which correspond to the type="story". The
    resulting text is sent to stdout. '''

import gzip
import sys
from pathlib import Path

from lxml import etree

if __name__ == "__main__":

    filenames = [Path(filename) for filename in sys.argv[1:]]

    for filename in filenames:
        with gzip.open(filename) as f:
            tree = etree.parse(f)
            docs = tree.xpath("DOC")
            for doc in docs:
                if doc.get("type") == "story":
                    p_s = doc.xpath("TEXT/P")
                    for p in p_s:
                        print(p.text)


