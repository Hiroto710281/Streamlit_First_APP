import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import time
import matplotlib.pyplot as plt

st.title('Streamlit_First')
st.write('Image')

if st.checkbox('Show Image'):
    img = Image.open('IMG_3587のコピー.png')
    st.image(img, caption='HIROTO', use_column_width=True)

"""
# 大学生です。　よろしくおねがいします。
- [Github](https://github.com/Hiroto710281)
##↓今の心情を表しています。↓
"""

# df = pd.DataFrame(
#     np.random.rand(100,2)/[50,50] + [35.69, 139.70],
#     columns=['lat','lon']
# )
# st.map(df)

df = pd.DataFrame(
np.random.rand(20, 3),
columns = ['a','b','c']

)
st.area_chart(df)

"""
###自分の出身地はだいたいこのあたりです。港北ニュータウンで有名です。
"""
df = pd.DataFrame(
    np.random.rand(100,2)/[50,50] + [35.5465, 139.5789],
    columns=['lat','lon']
)
st.map(df)


option = st.selectbox(
    'あなたが好きな数字を教えて下さい！！！！',
    list(range(1,11))
)
'あなたの好きな数字は', option, 'です'


text = st.text_input('あなたの趣味を教えてください')
'あなたの趣味',text,'です'

condition = st.slider('あなたの調子は？', 0, 100, 50)
condition




yuhi = 'src/yuhi02.csv'
yakei = 'src/yakei02.csv'
kaisui = 'src/kaisui02.csv'

@st.cache
def load_data(file):
    df_load = pd.read_csv(file,encoding='shift-jis')
    df1 = df_load.rename(columns={'北緯':'lat','東経':'lon'})
    df2 = pd.concat([df1,df1['所在地'].str.extract('(?P<都道府県>...??[都道府県])(?P<市区町村>.*)', expand=True)], axis=1).drop('所在地', axis=1)
    return df2

def main():
    st.title('日本のドライブで寄れそうなスポット')
    block_list = ['夕陽', '夜景', '海水']
    control_features = st.sidebar.selectbox('何を楽しみたい？',block_list)
    st.header(f'{control_features}100選')
    if control_features == '夕陽':
        visualize(yuhi)
    elif control_features == '夜景':
        visualize(yakei)
    elif control_features == '海水':
        visualize(kaisui)

def visualize(file):
    df = load_data(file)
    st.dataframe(df[['名称','都道府県','市区町村']])
    if st.sidebar.checkbox('mapを表示'):
        st.subheader('map')
        st.map(df)
    if st.sidebar.checkbox('都道府県ごとのグラフを表示'):
        st.subheader('都道府県ごとのグラフ')
        selected_prefectures = st.multiselect('都道府県を選択',df['都道府県'].unique().tolist(),['北海道','秋田県','青森県','富山県'])
        n = [(df['都道府県'] == prefecture).sum() for prefecture in selected_prefectures]
        # 棒グラフで表示
        sns.set(font="IPAexGothic") 
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        x = np.array(selected_prefectures)
        y = np.array(n)
        x_position = np.arange(len(x))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(x_position, y, tick_label=x)
        ax.set_xlabel('都道府県')
        ax.set_ylabel('数')
        st.pyplot(fig)

if __name__ == "__main__":
    main()























latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i +1)
    time.sleep(0.05)

'Thank You!!!!!!!!'

expander = st.beta_expander('Open!!!')
expander.write('A')
expander.write('B')
expander.write('C')
expander.write('D')




@st.cache
def sample():
    return pd.DataFrame(np.random.randint(0, 10, (6, 3)), columns=["A", "B", "C"])


df = sample()
title = st.sidebar.text_input("Title")  # グラフのタイトル
dc = {"line": st.line_chart, "bar": st.bar_chart, "area": st.area_chart}
kind = st.sidebar.radio("Kind", list(dc))  # ラジオボタン
memo = st.sidebar.text_area("Memo")  # メモ
number = st.sidebar.number_input("Number")
date = st.sidebar.date_input("Date")
time = st.sidebar.time_input("Time")
color = st.sidebar.color_picker("Color")

st.header(title)
dc[kind](df)

f"""
### Memo
{memo}
### Number
{number}
### Date
{date}
### Time
{time}
### Color
{color}
"""























