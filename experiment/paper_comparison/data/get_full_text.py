import json
import re
import os
import shutil
import csv
import glob
import gzip
import time
import boto3

_selfRefRE = r'((((our)|(this)|(proposed?)) (current)?((work)|(paper)|((([^\s]+) )?approach)|((([^\s]+) )?method)|(goals?)|((([^\s]+) )?focus)|((([^\s]+) )?system)|(research)|(results?)|(findings?)|(study)|(participants?)|(investigations?)|(evaluation)|(interview(s|(ees))?)))|(we instead)|(we propose)|(instead, we))'
selfRefRE = re.compile(_selfRefRE, re.I)

# Modified from 'https://github.com/allenai/RelatedlyAPI/blob/main/buildDataSet/200ParseS2ORC.py'
def parse(s2orc):
    out = {
        'paragraphs': [],
        'refs': [],
        'paper': None
    }
    def makeSegTuples(segs, kind, withBibDict=False):
        segs = json.loads(segs)
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
        bib_entries = json.loads(bib_entries)
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

    try:
        openAccess = s2orc['content']['source']['oa_info']['status'] or 'Unknown'
    except:
        pass
    license = 'Unknown'
    try:
        license = s2orc['content']['source']['oa_info']['license'] or 'Unknown'
    except:
        pass

    out['paper'] = {
        'corpusId': corpusId,
        'openAccess': openAccess,
        'license': license,
        'abstract': abstract
    }

    bib2corpusId = makeBib2CorpusId(grobid['annotations']['bib_entry'])

    body = grobid['contents']
    paragraphSegs = makeSegTuples(grobid['annotations']['paragraph'], 'paragraph')
    sectionSegs = makeSegTuples(grobid['annotations']['section_header'], 'section')
    bibSegs = makeSegTuples(grobid['annotations']['bib_ref'], 'bib', bib2corpusId)

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

        if referencedSelf:
            currents['refs'].sort(key=lambda ref: ref['start'])

        out['refs'].append(refs)
        out['paragraphs'].append({
            'corpusId': corpusId,
            'sectionTitle': currents['section'],
            'spans': json.dumps(currents['refs']),
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

            bib = {'corpusId': context, 'span': span, 'start': start, 'end': send}
            currents['refs'].append(bib)

            continue
    return out

def download_s2orc(corpus_id, aws_folder='s3://ai2-s2/paos/tmp_s2_newsletter'):
    # Clean up temporary folder
    os.system(f'aws s3 rm --recursive {aws_folder}')
    shutil.rmtree('tmp', ignore_errors=True)

    query = f'''
    UNLOAD
    (
        SELECT *
        FROM s2orc_papers.latest
        WHERE id = {corpus_id}
    )
    TO '{aws_folder}'
    WITH (format='json', compression='gzip')
    '''

    session = boto3.Session()
    athena = session.client('athena', 'us-west-2')
    execution = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': 's2orc_papers'
        },
        ResultConfiguration={
            'OutputLocation': aws_folder,
        }
    )

    execution_id = execution['QueryExecutionId']
    max_execution = 60

    state = 'RUNNING'
    while (max_execution > 0 and state in ['RUNNING', 'QUEUED']):
        max_execution = max_execution - 1
        response = athena.get_query_execution(QueryExecutionId = execution_id)

        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            if state == 'FAILED':
                print('FAILED', response)
            elif state == 'SUCCEEDED':
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]
                print('SUCCEEDED', filename)
        time.sleep(1)
    os.makedirs('tmp', exist_ok=True)
    os.system(f'aws s3 sync {aws_folder} tmp')
    index_file = glob.glob('tmp/*.csv')[0]
    s2orc = []
    with open(index_file) as filelist:
        files = csv.reader(filelist)
        for f in files:
            filename = f[0].split('/')[-1]
            with gzip.open(f'tmp/{filename}', mode='rt') as _f:
                s2orc.append(json.load(_f))
    return s2orc

def get_citing_paragraphs(cited_corpus_id, citing_corpus_id, citances=None, s2orc=None):
    if citances is None:
        citances = []
    if s2orc is None:
        s2orc = download_s2orc(citing_corpus_id)
    if len(s2orc) == 0:
        return None
    else:
        paragraphs = parse(s2orc[0])['paragraphs']
        citing_paragraphs = []
        for p in paragraphs:
            citances_detail = []
            for c in citances:
                if c in p['text']:
                    citances_detail.append({
                        'text': c,
                        'start': p['text'].find(c),
                        'end': p['text'].find(c) + len(c)
                    })
            if cited_corpus_id in [int(s['corpusId']) for s in json.loads(p['spans'])] or len(citances_detail) > 0:
                p['citances'] = citances_detail
                citing_paragraphs.append(p)
        return citing_paragraphs

def get_introduction_paragraphs(s2orc):
    if len(s2orc) == 0:
        return None
    else:
        paragraphs = parse(s2orc[0])['paragraphs']
        intro_paragraphs = []
        for p in paragraphs:
            if p['sectionTitle'].strip().lower() == 'introduction':
                intro_paragraphs.append(p)
        return intro_paragraphs

def get_related_work_paragraphs(corpus_id):
    s2orc = download_s2orc(corpus_id)
    if len(s2orc) == 0 or s2orc[0] is None:
        return None
    else:
        if parse(s2orc[0]) is None:
            return None
        paragraphs = parse(s2orc[0])['paragraphs']
        rw_paragraphs = []
        for p in paragraphs:
            if p['sectionTitle'].strip().lower() == 'related work' or 'background' in p['sectionTitle'].strip().lower() or 'related work' in p['sectionTitle'].strip().lower():
                rw_paragraphs.append(p)
        return rw_paragraphs
