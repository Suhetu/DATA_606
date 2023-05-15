from deta import Deta

DETA_KEY = 'b0y7a81l_pL1mNzFvgQh9ywWFcsG35tv8uhvXFDVH'

deta = Deta(DETA_KEY)

db = deta.Base('users_db')

def insert_user(username, name, password):

    return(db.put({'key': username, 'name': name, 'password': password}))

def fetch_all_users():

    res = db.fetch()
    return(res.items)

def get_user(username):

    return(db.get(username))

def update_user(username, updates):

    return(db.update(updates, username))

def delete_user(username):

    return(db.delete(username))
