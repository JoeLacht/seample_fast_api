import requests
import streamlit as st

st.title("Simple FastAPI application")

tab1, tab2 = st.tabs(['Image', 'Text'])

def main():
    with tab1:
        # create input form
        image = st.file_uploader("Классифицируй погоду", type=['jpg', 'jpeg'])
        if st.button("Определить!") and image is not None:
            # show image
            st.image(image)
            # format data for input format
            files = {"file": image.getvalue()}
            # send data and get the result
            # res = requests.post("http://127.0.0.1:8000/clf_image", files=files).json() # для запуска локально
            res = requests.post("http://backend:8000/clf_image", files=files).json() # для запуска через docker compose
            # print results
            st.write(f'Class name: {res["class_name"]}, class index: {res["class_index"]}')

    with tab2:
        txt = st.text_input('Классифицируй оценку отзыва на ресторан')
        if st.button('Определить'):
            text = {'text' : txt}
            # res = requests.post("http://127.0.0.1:8000/clf_text", json=text) # для запуска локально
            res = requests.post("http://backend:8000/clf_text", json=text)#.json() для запуска через docker compose
            st.write(res.json()['label'])
            st.write(res.json()['probability'])

if __name__ == '__main__':
    main()