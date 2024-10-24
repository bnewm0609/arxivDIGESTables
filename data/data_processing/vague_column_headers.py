vague_column_headers = """\
Venue
Month
Year
No.
[HTML]D0CECE\\nYear
Title
URL
[HTML]BBDAFFYear
Link
Version
Organiser
Citations
Pub.
Title of Survey Article
License
Publication
Repositories
Pub.
Code repository
First author
Paper Title
Published In
Team affiliations
Google Scholar citations (as of June 2022)
Publi. year
Article
Reference
Years
Journal
Conference
Stars / Citations
Papers
Authors [Ref]
Publication& Year
HuggingFace Repository Name
Available Link
Access Date
Ref.
# Cit.
Public Code
Published in
Year/Month
Authors
Book Title
#
Link to the code
Titles
Link of the corpus
Code Repository
ID
year
Publication year
NO.
Where
When
Github Link
Project Page
Repositories
First Author
Citations
Implementation
Venue Name
Citations
Publication Year/Type
Paper Title
Affiliation
Implementation
Webpage
Code Link
Publisher
Available Link
Commit
Title of articles
References count
Researchers
No. of papers
Github
Ref., Year
Publicly Available Repository (Data or Code)
license
Project Page
Conference/Journal
Abstract""".splitlines()

vague_column_headers = [ch.strip() for ch in vague_column_headers]

possibly_vague_column_headers = "Sources,Dataset source,Rank,ID,Sources,Nb.,Idx".split(",")