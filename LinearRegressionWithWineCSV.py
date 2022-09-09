from numpy.core.fromnumeric import argmin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy

# Hàm đọc file csv
def readFile(filename):
    df=pd.read_csv(filename,sep=";")
    return df


# Hàm trộn các dòng dữ liệu lại phục vụ cho câu 2
def shuffleMatrix(dataFrame):
    # để reset index để cho thứ tự đánh số dòng của dataframe không bị đổi theo
    df = dataFrame.sample(frac=1).reset_index(drop=True)
    return df

def processOneCol(df):
    # Chia số lượng dòng thành 5 phần bằng nhau
    splitPart=len(df)/5
    # Tạo mảng để chứa giá trị r trả về trong 5 lần lặp
    rTest=[]
    for i in range(1,6):
        # Bắt đầu từ 0 nên splitPart: 240 thì bắt đầu là 240*(i-1)=240*0=0
        begin=int(splitPart*(i-1))
        # Kết thúc cứ lấy splitPart*i
        end=int(splitPart*i)

        #print(df.iloc[begin:end])
        # Cột thứ nhất 0 là cột thuộc tính, lấy từ dòng begin đến end để đúng phần split
        attributeDF=df[df.columns[0]].iloc[begin:end]
        # Vì đây là ma trận chỉ có 1 cột n dòng nên ta phải reshape lại để không bị lỗi khi nhân ma trận
        attributeDF=attributeDF.values
        attributeDF=attributeDF.reshape(len(attributeDF),1)
        # Cột thứ hai 1 là cột quality, lấy từ dòng begin đến end để đúng phần split
        qualityDF=df[df.columns[1]].iloc[begin:end]

        # Gán thêm cột 1 vào trước cột thuộc tính cần tìm
        A,y=preprocess(attributeDF,qualityDF)
        # Tính x
        x=ols_linear_regression(attributeDF,qualityDF)
        # Tính phần dư r của phần thứ i trong cột thuộc tính gửi vào
        valueR=normR(attributeDF,qualityDF,x)

        # Nhét vào ma trận r chứa các giá trị r của 5 phần của 1 thuộc tính
        rTest.append(valueR)
    
    # Tính trung bình giá trị r của thuộc tính
    avgAtt=0
    for i in range(len(rTest)):
        avgAtt=avgAtt+rTest[i]
    return avgAtt/len(rTest)

            


def processQuestion2(df):
    #Tạo một mảng chứa giá trị r trả về
    rMatrix=[]
    # Chạy vòng lặp xét 11 cột dữ liệu không tính cột quality
    for i in range(0,11):
        rMatrix.append(processOneCol(df.iloc[:,[i,11]]))
    print("Sau khi tinh r 11 thuoc tinh: ")
    print(rMatrix)

    print("Gia tri phan du r nho nhat: ",min(rMatrix))
    print("Dac trung duy nhat dung de xay dung mo hinh la: ")
    print(df.columns[argmin(rMatrix)])

    # Vì chỉ có 1 thuộc tính --> 1 cột n dòng nên phải dùng cách reshape lại để tránh lỗi
    chosenCol=df.iloc[:,argmin(rMatrix)]
    chosenCol=chosenCol.values
    chosenCol=chosenCol.reshape(len(chosenCol),1)

    A,y=preprocess(chosenCol,df.iloc[:,11])
    result2=ols_linear_regression(A,y)

    print("Ket qua question 2 Theta: ")
    print(result2)



def processQuestion3(df):
    #Tạo một mảng chứa giá trị r trả về
    rMatrix=[]
    # Chạy vòng lặp xét 11 cột dữ liệu không tính cột quality
    for i in range(0,11):
        rMatrix.append(processOneCol(df.iloc[:,[i,11]]))

    print("Sau khi tinh r 11 thuoc tinh: ")
    print(rMatrix)

    # Tạo mảng rTemp là copy từ rMatrix giá trị 11 thuộc tính r
    rTemp=rMatrix.copy()
    # Đem 5 giá trị r nhỏ nhất lên đầu bằng cách xếp mảng từ nhỏ đến lớn
    for i in range(len(rTemp)):
        for j in range(i+1,len(rTemp)):
            if(rTemp[i]>rTemp[j]):
                temp=rTemp[i]
                rTemp[i]=rTemp[j]
                rTemp[j]=temp

    # Lấy 5 giấ trị r nhỏ nhất gán cho listFiveFactor
    listFiveFactor=rTemp[0:5]
    # Tạo list lưu lại index của 5 giá trị r vừa tìm được
    # Mục đích là để coi thử trong df tương ứng index đó là thuộc tính gì
    indexFiveFactor=[]

    # Chạy for append giá trị index cho ma trận 
    for i in listFiveFactor:
        indexFiveFactor.append(rMatrix.index(i))
    
    print("5 giá trị r nhỏ nhất sẽ được dùng: ")
    print(listFiveFactor)

    # Tạo list lưu lại tên các thuộc tính dùng để xây dựng mô hình dựa trên list index vừa tìm
    colNameFactor=[]
    for i in indexFiveFactor:
        colNameFactor.append(df.columns[i])
    
    print("Tên thuộc tính tương ứng với từng giá trị r: ")
    print(colNameFactor)
   
    A,y=preprocess(df.iloc[:,indexFiveFactor],df.iloc[:,11])
    result3=ols_linear_regression(A,y)
    print("Giá trị theta cho question 3: ")
    print(result3)

    r3=normR(A,y,result3)
    print("Phan du r la: ",r3)




# Hàm thêm cột 1 vào trước ma trận A trong Ax=b
def preprocess(x, y):
    A = np.hstack((np.ones((x.shape[0], 1)), x))
    return A, y

# hàm tính trọng số x
def ols_linear_regression(A, b):
    A_pinv = np.linalg.inv(A.T @ A) @ A.T    # np.linalg.pinv(A)
    b=b.values
    b=b.reshape(len(b),1)
    x = A_pinv @ b

    return x


# Tính tích trong của các phần tử trong 1 vector
def calc_inner_product(v, w):
    return sum(vi*wi for vi, wi in zip(v, w))

# Tính chuẩn v
def norm_square(v):
    return calc_inner_product(v, v)

# Hàm tính phần dư r
def normR(A,b,x):
    b=b.values
    b=b.reshape(len(b),1)
    
    r=norm_square(A@x-b)
    r=sympy.sqrt(r[0])
    return r


def main():
    df=readFile(filename="wine.csv")

    # cau 1 Sử dụng 11 đặc trưng đề bài cung cấp
    print(" *** Cau 1 *** : ")
    A,y=preprocess(df.iloc[:,0:11],df.iloc[:,11])
    result1=ols_linear_regression(A,y)
    print("Giá trị theta cho question 1: ")
    print(result1)

    r1=normR(A,y,result1)
    print("Phan du r la: ",r1)

    # cau 2 Tìm 1 đặc trưng duy nhất cho mô hình
    print("*** Cau 2 *** : ")
    df2=shuffleMatrix(df)
    processQuestion2(df2)

    # cau 3 Xây dựng mô hình của riêng bản thân
    print("*** Cau 3 *** : ")
    df3=shuffleMatrix(df)
    processQuestion3(df3)
    
main()