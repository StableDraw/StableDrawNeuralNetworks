from Delete_background import Delete_background
import os


if __name__ == '__main__':
    params = {
        "RescaleT": 320
    }
                    
    for root, dirs, files in os.walk("C:\\Users\\Robolightning\\Desktop\\1"):  
        for filename in files:
            print(filename)
            with open("C:\\Users\\Robolightning\\Desktop\\1\\" + filename, "rb") as f:
                init_img_binary_data = f.read()
            w, h, binary_data = Delete_background(init_img_binary_data, params) #передаю путь к рабочей папке и имя файла
            with open("C:\\Users\\Robolightning\\Desktop\\2\\" + filename, "wb") as f:
                f.write(binary_data)