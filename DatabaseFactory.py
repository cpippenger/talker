import json
import logging
from psycopg import connect,OperationalError
class DatabaseFactory:
    def __init__(self,debug=False):
        self._debug=debug
        self._conn=None
        self.connect()

    def _error(self,err:Exception)->None:
        print ("\nDataFactory ERROR:", err)
        return None
    
    def _query(self, sqlquery:str, sqldata:list) -> dict:
        logging.info(f"{__class__.__name__}._query(): {sqlquery=}, {sqldata=}}")
        data = {}
        try:
            cursor = self._conn.cursor()
            cursor.execute(sqlquery, sqldata)
            self._conn.commit()
            if "update" or "insert" not in sqlquery: # just for getting 1 object
                rowdata = cursor.fetchone()
                i = 0
                for desc in cursor.description:
                    data[desc[0]] = rowdata[i] # better way?
                    i=i+1
            else: 
                cursor.fetchone()
            cursor.close()
            
        except OperationalError as err:
            self._error(err)
        return data
    
    # this will save the child object to the database to a table based
    # on its classname. it will insert on save, use update for updating 
    # existing data it will save any varible in the child object that 
    # doesnt have an underscore in the name
    def save(self)->bool:
        savedata=self._getdata() 
        self._query(self._sqlstatement("insert",savedata),list(savedata.values()))
        return True

    # get data from object ignoring any underscores because they are "private"
    def _getdata(self)->dict:
        data={}
        for value in self.__dict__:
            if '_' not in value:
                data[value]=str(self.__dict__[value])         
        return data
    
    def _setdata(self,data):
        for key in data:
            value=str(data[key]).rstrip(" ") # this is database type hackery
            try: # detect json
                setattr(self,key,json.loads(value))
            except:
                setattr(self,key,value)

    #this will load an object by their uuid in the database, it uses the class name as the table
    def load(self,databaseid:str)->object:
        self.id=databaseid
        self._setdata(self._query(self._sqlstatement("select"),{"id":databaseid}))
        return True
    
    def _sqlstatement(self,statementtype="select",sqldata:dict=None)->str:
        if sqldata is None: # get data if not suppiled for the lazy
            sqldata=self._getdata()
        tablename=self.__class__.__name__
        columnnames=",".join(sqldata)
        if "select" in statementtype:
                return "select " + columnnames + " from \"" + tablename + "\" where id='"+ sqldata['id'] +"'"
        elif "insert" in statementtype:
                return "insert into \"" + tablename + "\" (" + columnnames + ") values (" + (" %s, " * len(sqldata) )[:-2] + ")"
        
        raise Exception("Unsupported Query Type")
        

    def connect(self,conn=None,host="127.0.0.1",port="5858",username="postgres",password="xxxPASSWORDxxx",database="talker") -> bool:
        try:
            self._conn= connect(
                dbname = database,
                user = username,
                host =host,
                password =password
            )
            status=True
        except OperationalError as err:
            self._error(err)
            status=False
        return status
        

