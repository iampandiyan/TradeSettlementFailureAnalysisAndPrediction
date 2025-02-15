import pandas as pd
import numpy as np

class AnalysisUtils():
   
    def quanQual(self,dataset):
        quan=[]
        qual=[]
        for columnName in dataset.columns:
            if(dataset[columnName].dtype=='O'):
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan,qual
    
    def createDescriptiveTable(self,dataset):
        quanColumns=self.quanQual(dataset)[0]
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR","1.5rule","Lesser","Greater","Min","Max","Kurtosis","Skew","Var","Std"],columns=quanColumns)        
        for columnName in quanColumns:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0]
            descriptive.loc["Q1:25%",columnName]=dataset.describe()[columnName]["25%"]
            descriptive.loc["Q2:50%",columnName]=dataset.describe()[columnName]["50%"]
            descriptive.loc["Q3:75%",columnName]=dataset.describe()[columnName]["75%"]
            descriptive.loc["99",columnName]=np.percentile(dataset[columnName],99)
            descriptive.loc["Q4:100%",columnName]=dataset.describe()[columnName]["max"]
            descriptive.loc["IQR",columnName]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
            descriptive.loc["1.5rule",columnName]=1.5*descriptive[columnName]["IQR"]
            descriptive.loc["Lesser",columnName]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5rule"]
            descriptive.loc["Greater",columnName]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5rule"]
            descriptive.loc["Min",columnName]=dataset[columnName].min()
            descriptive.loc["Max",columnName]=dataset[columnName].max()
            descriptive.loc["Kurtosis",columnName]=dataset[columnName].kurtosis()
            descriptive.loc["Skew",columnName]=dataset[columnName].skew()
            descriptive.loc["Var",columnName]=dataset[columnName].var()
            descriptive.loc["Std",columnName]=dataset[columnName].std()
        return descriptive
    
    def getOutliers(self,dataset):
        lesser=[]
        greater=[]
        quanColumns=self.quanQual(dataset)[0]
        descriptive=self.createDescriptiveTable(dataset)
        for columnName in quanColumns:
            if(descriptive[columnName]["Min"]<descriptive[columnName]["Lesser"]):
                lesser.append(columnName)
            if(descriptive[columnName]["Max"]>descriptive[columnName]["Greater"]):
                greater.append(columnName)
        return lesser,greater
        
    def removeOutlier(self,dataset):    
        descriptive=self.createDescriptiveTable(dataset)
        lesserCoulmNames=self.getOutliers(dataset)[0]
        greaterColumnNames=self.getOutliers(dataset)[1]
        for columnName in lesserCoulmNames:
            dataset.loc[dataset[columnName] < descriptive[columnName]["Lesser"], columnName] = descriptive[columnName]["Lesser"]
        for columnName in greaterColumnNames:
            dataset.loc[dataset[columnName] > descriptive[columnName]["Greater"], columnName] = descriptive[columnName]["Greater"]

    def createFrequencyTable(self,columnName,dataset):
        freqTable=pd.DataFrame(columns=["Unique_values","Frequency","Relative_Frequency","Cumsum"])
        freqTable["Unique_values"]=dataset[columnName].value_counts().index
        freqTable["Frequency"]=dataset[columnName].value_counts().values
        freqTable["Relative_Frequency"]=(freqTable["Frequency"]/103)
        freqTable["Cumsum"]=freqTable["Relative_Frequency"].cumsum()
        return freqTable
    
    def get_pdf_probability(dataset,startRange,endRange):
        from matplotlib import pyplot
        from scipy.stats import norm
        import seaborn as sns
        ax=sns.distplot(dataset,kde=True,kde_kws={'color':'blue'},color='Green')
        pyplot.axvline(startRange,color='Red')
        pyplot.axvline(endRange,color='Red')
        #generate a sample
        sample=dataset
        sample_mean=sample.mean()
        sample_std=sample.std()
        print('Mean=%.3f, Standard Deviation:%.3f' % (sample_mean,sample_std))
        #define distribution
        dist=norm(sample_mean,sample_std)
        #Sample probabilities
        values=[value for value in range(startRange,endRange)]
        probabilities=[dist.pdf(value) for value in values]
        prob=sum(probabilities)
        print("The Area between range ({},{}) is {}".format(startRange,endRange,sum(probabilities)))
        return prob
        
    def stdNormalDistributionGraph(dataset):
        import seaborn as sns
        mean=dataset.mean()
        std=dataset.std()
        values=[i for i in dataset]
        z_score=[((j-mean)/std) for j in values]
        sns.distplot(z_score,kde=True)