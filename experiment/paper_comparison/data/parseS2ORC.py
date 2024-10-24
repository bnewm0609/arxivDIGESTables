import re
import json

_selfRefRE = r'((((our)|(this)|(proposed?)) (current)?((work)|(paper)|((([^\s]+) )?approach)|((([^\s]+) )?method)|(goals?)|((([^\s]+) )?focus)|((([^\s]+) )?system)|(research)|(results?)|(findings?)|(study)|(participants?)|(investigations?)|(evaluation)|(interview(s|(ees))?)))|(we instead)|(we propose)|(instead, we))'
selfRefRE = re.compile(_selfRefRE, re.I)

def parse(s2orc):
    global selfRefParas
    global selfRefCounts

    out = {
        'paragraphs': [],
        'refs': [],
        'paper': None
    }

    def makeSegTuples(segs, kind, withBibDict=False):
        #segs = json.loads(segs)
        if withBibDict != False:
            tuples = []
            for seg in segs:
                if not 'attributes' in seg:
                    continue
                start, end, bib = seg['start'], seg['end'], seg['attributes']['ref_id']
                if not bib in withBibDict:
                    continue
                cid = withBibDict[bib]
                tuples.append((start, end, kind, cid))
            return tuples
        return [(seg['start'], seg['end'], kind, None) for seg in segs]

    def makeBib2CorpusId(bib_entries):
        #bib_entries = json.loads(bib_entries)
        bib2corpusId = {}
        for entry in bib_entries:
            attrs = entry['attributes']
            if not 'matched_paper_id' in attrs:
                continue
            bid = attrs['id']
            cid = attrs['matched_paper_id']
            bib2corpusId[bid] = cid
        return bib2corpusId

    content = s2orc['content']
    if not content:
        return None

    grobid = content['grobid']
    if (not grobid) or \
            (not grobid['annotations']['bib_entry']) or \
            (not grobid['annotations']['paragraph']) or \
            (not grobid['annotations']['section_header']) or \
            (not grobid['annotations']['bib_ref']):
        return None

    corpusId = s2orc['id']
    abstract = s2orc['metadata']['abstract']
    openAccess = 'Unknown'

    title = s2orc['metadata']['title']
    conference = s2orc['metadata']['venue'] if "venue" in s2orc['metadata'] else 'Unknown'
    year = s2orc['metadata']['publication_date']['year'] # the `year` field is s2orc process year

    try:
        openAccess = s2orc['content']['source']['oa_info']['status'] or 'Unknown'
    except:
        pass
    license = 'Unknown'
    try:
        license = s2orc['content']['source']['oa_info']['license'] or 'Unknown'
    except:
        pass


    print('%s:%s' % (corpusId, title))

    out['paper'] = {
        'corpusId': corpusId,
        'title': title,
        'conference': conference,
        'year': year,
        'openAccess': openAccess,
        'license': license,
        'abstract': abstract
    }

    bib2corpusId = makeBib2CorpusId(grobid['annotations']['bib_entry'])

    body = grobid['contents']
    paragraphSegs = makeSegTuples(grobid['annotations']['paragraph'], 'paragraph')
    sectionSegs = makeSegTuples(grobid['annotations']['section_header'], 'section')
    bibSegs = makeSegTuples(grobid['annotations']['bib_ref'], 'bib', bib2corpusId)
    
    print(len(paragraphSegs))

    allSegs = paragraphSegs + sectionSegs + bibSegs
    allSegs.sort()

    def makeCurrents():
        return {
            'section': 'Init',
            'paragraph': None,
            'refs': []
        }

    currents = makeCurrents()

    def emit(currents):
        global selfRefParas
        global selfRefCounts
        refs = set(map(lambda e: e['corpusId'], currents['refs']))
        size = len(refs)
        '''
        print('='*33)
        print(currents['section'])
        print('-'*33)
        print(currents['paragraph'][1])
        print(currents['refs'])
        print((size, len(currents['refs'])))
        print('='*33)
        '''

        if currents['paragraph'] == None:
            return

        likelyRelatedWorkSection = currents['section'] != 'Init' and \
            not 'introduction' in currents['section'].lower() and \
            not 'discussion' in currents['section'].lower() and \
            not 'future work' in currents['section'].lower() and \
            not 'future direction' in currents['section'].lower()

        text = currents['paragraph'][1]
        selfRefs = list(selfRefRE.finditer(text))
        referencedSelf = False
        for selfRef in selfRefs:
            referencedSelf = True
            refs.add(corpusId)
            start = selfRef.span()[0]
            end = selfRef.span()[1]
            currents['refs'].append({
                'corpusId': corpusId, 'span': text[start:end], 'start': end, 'end': end
            })

        if len(refs) >= 3:

            if referencedSelf:
                currents['refs'].sort(key=lambda ref: ref['start'])

            out['refs'].append(refs)
            out['paragraphs'].append({
                'corpusId': corpusId,
                'title': title,
                'sectionTitle': currents['section'],
                'spans': json.dumps(currents['refs']),
                'conference': conference,
                'year': year,
                'likelyRelatedWorkSection': likelyRelatedWorkSection,
                'text': text,
                'refCount': len(refs)
            })
            if referencedSelf and False:
                pprint(out['paragraphs'][-1])
                pprint(selfRefs)
                print('='*33)

    for seg in allSegs:
        start, end, kind, context = seg
        start = int(start)
        end = int(end)
        seg = (start, end, kind, context)


        text = body[start:end]

        if kind == 'section':
            emit(currents)
            currents = makeCurrents()

            currents['section'] = text
            continue

        if kind == 'paragraph':
            emit(currents)
            oldSection = currents['section']
            currents = makeCurrents()
            currents['section'] = oldSection

            currents['paragraph'] = (seg, text)
            continue

        if kind == 'bib':
            if currents['paragraph'] == None:
                continue

            paragraph = currents['paragraph'][1]
            pstart = currents['paragraph'][0][0]
            sstart, send = ((start-pstart), (start-pstart+end-start))
            span = paragraph[sstart:send]

            bib = {'corpusId': context, 'span': span, 'start': sstart, 'end': send}
            currents['refs'].append(bib)

            continue
    return out

# if __name__ == '__main__':
#     import sys
#     s2orc = json.load(open(sys.argv[1]))
#     out = parse(s2orc)
#     print(out)
#     import pdb; pdb.set_trace()

if __name__ == '__main__':
    import sys
    with open('/Users/yjlee/Desktop/KIXLAB/NewsLetter/data/promptiverse.json', 'r') as file:
        s2orc = json.load(file)

    # s2orc = json.load('data/promptiverse.json')
    out = parse(s2orc)
    print(out)
    import pdb; pdb.set_trace()
    file_path = '/Users/yjlee/Desktop/KIXLAB/NewsLetter/data/tmp_parsed.json'
    with open(file_path, 'w') as file:
        json.dump(out, file)
        
    # print(out)
    # import pdb; pdb.set_trace()
