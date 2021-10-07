import numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, config, train, test) -> None:
        self.config = config

        self.train = train
        # print("self.train.head():", self.train.head())
        self.test = test
        # print("self.test.head():", self.test.head())

    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(self, ticket ):
        ticket = ticket.replace( '.' , '' )
        ticket = ticket.replace( '/' , '' )
        ticket = ticket.split()
        ticket = map( lambda t : t.strip() , ticket )
        ticket = list(filter( lambda t : not t.isdigit() , ticket ))
        if len( ticket ) > 0:
            return ticket[0]
        else: 
            return 'XXX'
        

    def preprocess(self):
        full = self.train.append( self.test , ignore_index = True )
        # print("full.head():")
        # print(full.head())
        titanic = full[ :891 ]
        # print("titanic.head():")
        # print(titanic.tail(100))



        embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
        # print(embarked.head()) #creating dummy variable for each pessneger


        # Create dataset
        imputed = pd.DataFrame()
        imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )
        imputed[ 'Age' ] = (imputed[ 'Age' ] - imputed[ 'Age' ].min())/ (imputed[ 'Age' ].max() - imputed[ 'Age' ].min())
        # print("imputed:", imputed.min())


        # Fill missing values of Fare with the average of Fare (mean)
        # imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )
        imputed[ 'Fare' ] = (full.Fare.fillna( full.Fare.mean() ) - full.Fare.min())/( full.Fare.max() - full.Fare.min())


        title = pd.DataFrame()
        # we extract the title from each name
        title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )


        # a map of more aggregated titles
        Title_Dictionary = {
                            "Capt":       "Officer",
                            "Col":        "Officer",
                            "Major":      "Officer",
                            "Jonkheer":   "Royalty",
                            "Don":        "Royalty",
                            "Sir" :       "Royalty",
                            "Dr":         "Officer",
                            "Rev":        "Officer",
                            "the Countess":"Royalty",
                            "Dona":       "Royalty",
                            "Mme":        "Mrs",
                            "Mlle":       "Miss",
                            "Ms":         "Mrs",
                            "Mr" :        "Mr",
                            "Mrs" :       "Mrs",
                            "Miss" :      "Miss",
                            "Master" :    "Master",
                            "Lady" :      "Royalty"

                            }

        # we map each title
        # df.column.method
        title[ 'Title' ] = title.Title.map( Title_Dictionary )
        title = pd.get_dummies( title.Title )
        #title = pd.concat( [ title , titles_dummies ] , axis = 1 )


        cabin = pd.DataFrame()

        # replacing missing cabins with U (for Uknown)
        cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )

        # mapping each Cabin value with the cabin letter
        cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

        # dummy encoding ...
        cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

        # print(cabin.head())


        ticket = pd.DataFrame()

        # Extracting dummy variables from tickets:
        ticket[ 'Ticket' ] = full[ 'Ticket' ].map( self.cleanTicket )
        ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

        print(ticket.head())

        family = pd.DataFrame()

        # introducing a new feature : the size of families (including the passenger)
        family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1


        family[ 'FamilySize' ] = (family[ 'FamilySize' ] - family[ 'FamilySize' ].min())/( family[ 'FamilySize' ].max() - family[ 'FamilySize' ].min())
        # print('family["FamilySize"]',family["FamilySize"].max())
        # print('family["FamilySize"]', family["FamilySize"].min())



        # # introducing other features based on the family size
        # family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
        # family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
        # family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

        # Transform Sex into binary values 0 and 1
        sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

        full_X = pd.concat( [ imputed , embarked , cabin, family , sex,  ] , axis=1 )

        # Create all datasets that are necessary to train, validate and test models
        train_valid_X = full_X[ 0:891 ]
        train_valid_y = titanic.Survived
        test_X = full_X[ 891: ]

        return train_valid_X, train_valid_y, test_X













        