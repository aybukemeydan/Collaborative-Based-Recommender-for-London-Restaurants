# Initialize
import flask
from flask import request, render_template
import pickle
import pandas as pd
import os
#import findspark
#findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import Pipeline as PL
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

app = flask.Flask(__name__,template_folder="templates")

result = pd.read_csv("files/new.csv")
result2 = pd.read_csv("files/spark2.csv")


url_alias = result[["name","cafe_links","cafe_types"]].drop_duplicates()

outlet_list = result2["name"].drop_duplicates()
url_alias = list(zip(url_alias["name"],url_alias["cafe_links"]))

@app.route('/')
def initial_ratings():
    return render_template('index.html', list=outlet_list, url_alias=url_alias)

@app.route('/', methods=["GET", "POST"])
def calculate():
    userid = 107487
    res1 = request.form['name1']
    rate1 = float(request.form['rate1'])
    res2 = request.form['name2']
    rate2 = float(request.form['rate2'])
    res3 = request.form['name3']
    rate3 = float(request.form['rate3'])
    res4 = request.form['name4']
    rate4 = float(request.form['rate4'])
    res5 = request.form['name5']
    rate5 = float(request.form['rate5'])

    #creating a new spark session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    path = "spark2.csv"
    df = spark.read.csv(path, header='true', inferSchema='true', sep=',') # 21 KB
    df = df.drop("_c0")
    vals = [(res1,rate1,userid),(res2,rate2,userid),(res3,rate3,userid),(res4,rate4,userid),(res5,rate5,userid)]
    new_item = spark.createDataFrame(vals,df.columns)
    df = df.union(new_item)
    pd_df = df.toPandas()
    stage1 = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(df.columns)-set(['rating']))] # string => number
    stage2 = ALS(userCol='userID_index',
          itemCol='name_index',
          ratingCol='rating', 
          nonnegative = True,
          implicitPrefs = False,
          coldStartStrategy="drop",
          maxIter= 10,
          regParam= 0.01,
          alpha=0.01,
          rank= 200,
          seed=42) # params after tuned RMSE : 1.71

    # setup the pipeline
    pipeline_try = Pipeline().setStages(stage1 + [stage2])
    model = pipeline_try.fit(df)
    t_df = model.transform(df)


    recs = model.stages[-1].recommendForAllUsers(7).toPandas() # alsmodel =  model.stages[-1]

    nrecs=recs.recommendations.apply(pd.Series) \
                .merge(recs, right_index = True, left_index = True) \
                .drop(["recommendations"], axis = 1) \
                .melt(id_vars = ['userID_index'], value_name = "recommendation") \
                .drop("variable", axis = 1) \
                .dropna()
    nrecs=nrecs.sort_values('userID_index')
    nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['userID_index']], axis = 1)
    nrecs.columns = [

            'name_index',
            'Rating',
            'UserID_index'

         ]

    md=t_df.select(t_df['userID'],t_df['userID_index'],t_df['name'],t_df['name_index'])
    md=md.toPandas()
    dict1=dict(zip(md['userID_index'],md['userID']))
    dict2=dict(zip(md['name_index'],md['name']))
    nrecs['UserID']=nrecs['UserID_index'].map(dict1)
    nrecs['name']=nrecs['name_index'].map(dict2)

    new=nrecs[['UserID','name','Rating']]
    new['recommendations'] = list(zip(new.name, new.Rating))

    res=new[['UserID','recommendations']]
    res_new=res['recommendations'].groupby([res.UserID]).apply(list).reset_index()
    collab_rec = pd.DataFrame(dict(res_new[res_new["UserID"]==userid]['recommendations'].tolist()[0]),index=[0]).T.sort_values(0,ascending=False)

    
    rated = pd_df[pd_df['userID']==userid]['name'].tolist()

   
    collab_rankedrecs = collab_rec.loc[[name for name in collab_rec.index if name not in rated],0]

    
    collab_df = pd.DataFrame({'recommendations':collab_rankedrecs.index,'collab_filter_predicted_ratings':collab_rankedrecs})
    top_5_recs = collab_df[['recommendations','collab_filter_predicted_ratings']].sort_values('collab_filter_predicted_ratings',ascending=False).head()
    top_5_recs.reset_index(drop=True,inplace=True)
    

    
    first = top_5_recs.loc[0,'recommendations']
    second = top_5_recs.loc[1,'recommendations']
    third = top_5_recs.loc[2,'recommendations']

    return render_template('pos.html', first=first, second=second, third=third, fourth="None", fifth="None", shop1=shop1, rate1=rate1, shop2=shop2, rate2=rate2, shop3=shop3, rate3=rate3, shop4=shop4, rate4=rate4, shop5=shop5, rate5=rate5,  url_alias=url_alias)


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
