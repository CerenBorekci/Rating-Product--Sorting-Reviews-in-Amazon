                                 # PROJE: Rating Product & Sorting Reviews in Amazon

"""
                                 İş Problemi
E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve
satın alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru 
bir şekilde sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan 
etkileyeceğinden dolayı hem maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret
sitesi ve satıcılar satışlarını arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

"""


# Veri Seti Hikayesi
# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


        # GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.

"""
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.
"""

#Öncelikle gerekli kütüphaneleri import edip veri setimizi çağıralım.

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\sevim\Desktop\MIUUL\HAFTA 4\CASE STUDY 1\Rating Product&SortingReviewsinAmazon\amazon_review.csv")


#Veri setimizi gözlemleyelim.
df.head()
df.info()
"""
#unixReviewTime      int64
reviewTime          object
day_diff            int64       tarihler dayse cevrilmeli.
"""
df.describe().T
df.shape  #(4915, 12)
df["reviewerID"].nunique()   #4915 >>> reviewerIDler unique, gruplandırarak tekilleştirmemize gerek yok.



# Adım 1:Ürünün Ortalama Puanını Hesaplayınız.
df["overall"].mean()   #4.587589013224822
#hiçbir işlem yapmadığımzda sistemin bize verdiği overall ortalaması 4.58 değerinde. Tarihe göre ağırlıklı ortalama
#ile arasındaki farkı inceleyeceğiz.

df["overall"].value_counts()
#hangi puandan ne kadar kullanıldığını hesapladık.

"""
overall
5.00000    3922
4.00000     527
1.00000     244
3.00000     142
2.00000      80
"""



df["overall"].describe()

"""
count   4915.00000
mean       4.58759
std        0.99685
min        1.00000
25%        5.00000
50%        5.00000
75%        5.00000
max        5.00000
"""

# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
# reviewTime: Değerlendirme zamanı Raw
#öncelikle müşterinin değerlendirme zamanının tipini değiştirmemiz gerekiyor, daha sonra bir değişken yaratıp müşterinin
#güncel analiz durumundan ne kadar önce yorum yaptığını hesaplamalıyız.

for i in df.columns:
    if "reviewTime" in i:
        df[i]=pd.to_datetime(df[i])

df["reviewTime"].max() ##2014-12-07 00:00:00
current_date = df["reviewTime"].max()
df["day_diff"] = (current_date - df["reviewTime"]).dt.days

#Yeni oluşturduğumuz değişkenin typeini inceleyelim.
df["day_diff"].dtypes  #dtype('int64')

df["day_diff"].head() #1063
df["day_diff"].mean() #436.3670396744659
df["day_diff"].describe().T

"""
count   4915.00000
mean     436.36704
std      209.43987
min        0.00000
25%      280.00000
50%      430.00000
75%      600.00000
max     1063.00000
"""
#Buradaki çeyrekliklerde bulunan değerler ile ağırlıklı ortalama belirleyeceğiz.

df.loc[df["day_diff"]<=280, "overall"].mean()         #4.6957928802588995

df.loc[(df["day_diff"]>280) & (df["day_diff"]<=430), "overall"].mean() #4.636140637775961

df.loc[(df["day_diff"]>430) & (df["day_diff"]<=600), "overall"].mean()  #4.571661237785016

df.loc[df["day_diff"]>600 , "overall"].mean()    #4.4462540716612375

#loc değişkeni ile kalıcı atama yaptık.

"""
#Müşterinin yorum yaptığı gün sayısı arttıkça vereceğimiz ağırlık azalacaktır. Örneğin,
#bir müşteri 600 gün önce yorum yaptıysa müşterinin ürünün populeritesini ve güncel durumdaki analizi için sağlıklı
#örneği bize veremeyebilir.Ağırlık arttıkça hareket edeceğimiz range azalır.
#Yüksek ağırlık en yakın tarihe verilir, sıcak müşteridir.
"""

#Şimdi ağırlıklarını belirleyeceğimiz bir fonksiyon yaratalım.


def time_based_weighted_average(dataframe, w1=30, w2=26, w3=24, w4=20):
    return dataframe.loc[df["day_diff"] <= 280, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 280) & (dataframe["day_diff"] <= 430), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 430) & (dataframe["day_diff"] <= 600), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 600), "overall"].mean() * w4 / 100
time_based_weighted_average(df)

#Buradaki önemli ayrıntı, ağırlıkların total 100 olması gerekir.Aşağıdaki gibi problem akışına göre değiştirebiliriz.
# time_based_weighted_average(df, 30, 26, 22, 22)



"""
ilk ortalama değerinde hicbir zaman dilimi gözekmeksizin ortalamalarını aldık. Bu değerlere baktığımızda en son yorum
yapan , veya düzenli olarak yorum yapan müsteriler arasında fark olmadıgını baz aldık.
Daha saglıklı olması acısından kullanıcıları zaman periyoduna böldük.Bu zaman periyoduna göre müşteri taleplerine göre
en sona yakın yorum yapanların yani day_diffi daha düsük olanların güncele daha yakın olacağından ağırlığı da yüksek 
olacak şekilde fonksiyon yarattık.
Böylece müşteri trendlerini de check edebiliyoruz. Buna göre ise beğeni ortalaması tarih agırlıklı normal averagetan daha
yüksek çıktı.

"""

#Normal ortalama : 4.587589013224822
#Ağırlıklı zaman ortalaması : 4.600583941300071


           # Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.

# Adım 1. helpful_no Değişkenini Üretiniz.


# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

#Elimizde total_vote değişkeni ve helpful_yes değişkeni bulunmaktadır. helpful_no değişkeni için total_vote ile helpful_yes
#arasındaki farkı almamız yeterli olacaktır.

"""
helpful_no = total_vote-helpful_yes olacak. 
total_vote= up+down
"""


df.head()
df["helpful_no"] =df["total_vote"]-df["helpful_yes"]

#Bizim için gereken indexleri seçerek dataframe üzerinde sadeleştirme yaptık.
df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]



# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
"""
• score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
• score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
• score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
• wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

"""

                                        ## score_pos_neg_diff fonk tanımlama
def score_pos_neg_diff(up,down):
    return up-down

df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"],df["helpful_no"])

df["score_pos_neg_diff"]=df["helpful_yes"]- df["helpful_no"]




                                         ##  score_average_rating fonk tanımlama

def score_average_rating(up,down):
    if up+down == 0:
        return 0
    else:
        return up / (up+down)


df["score_average_rating"]= df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


                                          #wilson_lower_bound fonk tanımlama

def wilson_lower_bound(up,down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()



# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
df.sort_values("wilson_lower_bound", ascending=False).head(20)


"""
Yorumların olumlu ve olumsuz olma dağılımlarına göre bir sıralama gerçekleştirdik. Bunu 3 metodla yapıp genel kıyaslamada
bulunduk. İlk olarak olumlu ve olumsuz yorumların farklarına göre, sonrasında olumlu yorum oranına göre ve en son
Wilson Lower Bound metoduna göre sıraladık. Örneklemdeki herhangi bir yanlılığı hesaba katmak için güven aralığının alt 
sınırını ayarlayan bir düzeltme faktörüdür. Wilson Lower Bound, bir veri örneğine sahip olduğumuz ve popülasyondaki 
gerçek oran hakkında ne kadar emin olabileceğimizi bilmek istediğimiz durumlarda kullanışlıdır diyebiliriz.
Ürünü en iyi temsil eden ilk 10 yoruma erişmiş olduk. Sıralamayı en yüksek puanlara göre değil ürünü en iyi temsil eden 
yorumlara göre oluşturduk. Çok daha objektif bir sıralama elde ettik.
"""

