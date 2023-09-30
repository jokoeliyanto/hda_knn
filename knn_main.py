import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_blobs

centers = 2
X, y = make_blobs(n_samples=700, centers=centers, random_state=20)


def input_converter(text_list):
  text_list = text_list.replace("[", "").replace("]", "").split(",")
  text_list_arr = np.array(text_list)
  text_list_arr = [eval(i) for i in text_list_arr]
  return text_list_arr

def jarak_euclidean(x,y):
    return np.sqrt(sum(pow(a-b, 2) for a, b in zip(x,y)))

def KNN(X,y, x_baru_arr):

    jarak=[]
    for x_i in X:
        jarak_euclid=jarak_euclidean(x_i,x_baru_arr)
        jarak.append(jarak_euclid)

    urut=np.sort(jarak)
    list_urut=[]
    for i in range(len(urut)):
        list_urut.append(urut[i])

    urutan=[]
    for i in list_urut:
        urtn=jarak.index(i)
        urutan.append(urtn)

    np.argsort(jarak)
    ambil=urutan[:k]
    radius = max(np.array(jarak)[ambil])
    kelas_paling_banyak = Counter(y[ambil]).most_common(1)
    kelas_data  = kelas_paling_banyak[0][0]
    return kelas_data, radius

def KNN_plot_1(X,y,x_baru_arr,kelas_data, radius):
    df = pd.DataFrame(X, columns =['x1', 'x2'])
    colormap = np.array(['g', 'b'])

    fig, ax = plt.subplots()
    ax.scatter(df['x1'], df['x2'], color=colormap[y], s=5)
    ax.scatter(x_baru_arr[0], x_baru_arr[1], color='r',  label="Data Baru", marker="s")
    ax.set_aspect( 1 )
    ax.set_title("Data Baru = {}".format(x_baru))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

    return fig

def KNN_plot_2(X,y,x_baru_arr,kelas_data, radius):
    df = pd.DataFrame(X, columns =['x1', 'x2'])
    colormap = np.array(['g', 'b'])

    data_baru_color = "r"
    if kelas_data==0:
        data_baru_color = "g"
    else: 
        data_baru_color = "b"

    fig, ax = plt.subplots()
    ax.scatter(df['x1'], df['x2'], color=colormap[y], s=5)
    ax.scatter(x_baru_arr[0], x_baru_arr[1], color=data_baru_color,  marker="s")
    k_circle = plt.Circle( (x_baru_arr[0], x_baru_arr[1] ),
                            radius,
                            fill = True,
                            alpha=0.2,
                            color="red")
    ax.set_aspect( 1 )
    ax.add_artist( k_circle )
    ax.set_title("K = {} , Data Baru = {}".format(k, x_baru))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    return fig


df = pd.DataFrame(X, columns =['x1', 'x2'])

st.title('Demo K - Nearest Neighbor(KNN)')
st.subheader("Hobi Data Academy")
st.text("Oleh : Joko Eliyanto & Indra Cahya Ramdani")

st.text("""
        
        Algoritma KNN adalah salah satu algoritma Machine Learning yang melakukan 
        klasifikasi data didasarkan pada data-data tetangganya. 

        """)

with st.sidebar:
    st.title("Parameter KNN")

    k = st.slider("Jumlah tetangga terdekat (K):", 
                min_value=1, 
                max_value=20, 
                value=3)

    x_baru = st.text_input("Data baru yang akan diklasifikasi [x,y]:",  
                value="[4.5,7]", 
                label_visibility="visible")
    
    tbl_knn = st.button(label="Klasifikasikan", type="primary")
    tbl_reset = st.button(label="Reset")

x_baru_arr = input_converter(x_baru) 



if tbl_knn:
    kelas_data, radius = KNN(X,y, x_baru_arr)
    fig = KNN_plot_2(X,y,x_baru_arr, kelas_data, radius)
    a = st.pyplot(fig)

elif tbl_reset:
    kelas_data, radius = KNN(X,y, x_baru_arr)
    fig = KNN_plot_1(X,y,x_baru_arr, kelas_data, radius)
    a = st.pyplot(fig)

else:
    kelas_data, radius = KNN(X,y, x_baru_arr)
    fig = KNN_plot_1(X,y,x_baru_arr, kelas_data, radius)
    a = st.pyplot(fig)


st.subheader("Algoritma KNN")
st.text("""
        
        1. Menentukan parameter k (jumlah tetangga paling dekat);
        2. Menghitung jarak euclidean objek terhadap data training yang diberikan;
        3. Mengurutkan hasil no 2 secara ascending (berurutan dari nilai 
           tertinggi ke terendah);
        4. Mengumpulkan kategori data (Klasifikasi K-Nearest Neighbour 
           berdasarkan nilai k); dan
        5. Dengan menggunakan kategori KNN yang paling mayoritas maka 
           dapat diprediksi kategori objek.

""")

st.subheader("Ayat & Hadist Terkait")
st.text("""
        
        “Bagaimana mungkin (tidak mungkin) kalian menjadi kafir, sedangkan 
        ayat-ayat Allah dibacakan kepada kalian, dan Rasul-Nya Pun berada 
        ditengah-tengah kalian? Dan barangsiapa yang berpegang teguh kepada 
        (agama) Allah maka sesungguhnya dia telah diberi 
        petunjuk kepada jalan yang lurus.” 
        (QS. Ali ‘Imran: 101).

        “Seseorang yang duduk (berteman) dengan orang sholih dan orang yang jelek 
        adalah bagaikan berteman dengan pemilik minyak misk dan pandai besi. 
        Jika engkau tidak dihadiahkan minyak misk olehnya, 
        engkau bisa membeli darinya atau minimal dapat baunya. 
        Adapun berteman dengan pandai besi, 
        jika engkau tidak mendapati badan atau pakaianmu hangus terbakar, 
        minimal engkau dapat baunya yang tidak enak.” 
        (HR. Bukhari No. 2101)

        """)
