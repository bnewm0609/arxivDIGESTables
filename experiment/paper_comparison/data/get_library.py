import psycopg2
import os

LIBRARY_FIELDS = [
    'library_folder_id',
    'name',
    'created_at_utc',
    'modified_at_utc',
    'archived_at_utc',
    'recommendation_status',
    'source_type',
    'description',
    'privacy_setting',
]

PAPER_FIELDS = [
    'entry_id',
    'created_at_utc',
    'modified_at_utc',
    'corpus_id',
    'paper_id',
    'paper_title',
    'annotation_state', # 'NotRelevant' means down vote
    'source_type',
    'status',
    'library_folder_id',
]

def get_db_connection():
    connection = psycopg2.connect(
        host='172.31.8.103',
        port='5439',
        database='dev',
        # user=os.environ['S2_REDSHIFT_USER'],
        # password=os.environ['S2_REDSHIFT_SECRET']
        user="johnconnor",
        password='6RL655Pi9U%#VeQj3Nmn2U^qm&79ZXV6'
    )
    return connection

def get_user_ids(email):
    connection = get_db_connection()
    cur = connection.cursor()
    cur.execute(
        f'''
        SELECT app_user_id
        FROM online.app_user
        WHERE email='{email}'
        '''
    )
    res = cur.fetchall()
    connection.close()
    return [r[0] for r in res]

def get_library_folders(app_user_id):
    connection = get_db_connection()
    cur = connection.cursor()
    cur.execute(
        f'''
        SELECT {','.join(LIBRARY_FIELDS)}
        FROM online.library_folder
        WHERE app_user_id={app_user_id}
        '''
    )
    res = cur.fetchall()
    connection.close()
    folders = []
    for r in res:
        folders.append(
            dict((k, v) for k, v in zip(LIBRARY_FIELDS, r))
        )
    return folders

def get_library_papers(app_user_id, folder_ids=None):
    connection = get_db_connection()
    cur = connection.cursor()
    folder_filter = '' if not folder_ids else f'''
        library_folder_id IN ({",".join([str(id) for id in folder_ids])}) AND
    '''
    cur.execute(
        f'''
        SELECT {','.join(PAPER_FIELDS)}
        FROM online.library_entry
        WHERE {
            folder_filter
        } app_user_id={app_user_id}
        '''
    )
    res = cur.fetchall()
    connection.close()
    papers = []
    for r in res:
        papers.append(
            dict((k, v) for k, v in zip(PAPER_FIELDS, r))
        )
    return papers
