import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
    if programPause == 'q':
        sys.exit()


def SA(search_space_tree, search_space_depth, train_images, train_labels, val_images, val_labels, T):
    ### Sıcaklığın kökünü alıyoruz ve scale değişkenine atanır.
    scale = np.sqrt(T)
    
    ### search_space_tree ve search_space_depth aralıklarından rastgele başlangıç noktaları seçilir.
    start_tree = np.random.choice(search_space_tree)
    start_depth = np.random.choice(search_space_depth)
    
    ### Baslangıç noktaları x_tree ve x_depth değişkenlerine atınır.
    x_tree = start_tree
    x_depth = start_depth
    
    
    ### Sınıflandırı x_tree ve x_depth parametrelerine göre oluşturulur.
    clf = RandomForestClassifier(n_estimators=int(x_tree), max_depth=int(x_depth), class_weight='balanced')
        
    ### Sınıflandırıcı eğitilir ve eğitim süresi hesaplanır.
    start_time = time.time()
    clf = clf.fit(train_images, train_labels)
    execution_time_previous = (time.time() - start_time)
    
    ### Sınıflandırıcı val veri seti için tahminde bulunur ve yanılma değeri hesaplanır.
    pred = clf.predict(val_images)
    accuracy = accuracy_score(val_labels, pred)
    previous_loss = 1 - accuracy
    
    ### kaybımız loss_graph listesine, eğitim süremiz time_graph listesine eklenir. 
    loss_graph = [previous_loss]
    time_graph = [execution_time_previous]
    
    ### Döngümüz başlar.
    for i in range(1000):
        ### seçilmiş x_tree ve x_depth noktasından rastgele bir şekilde uzaklaşılır ve new_x_tree, new_x_depth değişkenlerine atanır.
        new_x_tree = x_tree + np.random.normal() * search_space_tree[-1] / scale
        new_x_depth = x_depth + np.random.normal() * search_space_depth[-1] / scale
        ###new_x_tree = np.random.normal() * search_space_tree[-1]
        ###new_x_depth = np.random.normal() * search_space_depth[-1]
        
        ### eğer rastgele seçilen noktamız örneklem uzayının dışında ise sıradaki döngüye geçilir.
        if new_x_tree > search_space_tree[-1] or new_x_tree < search_space_tree[0] or int(new_x_tree) == 0:
            continue
        if new_x_depth > search_space_depth[-1] or new_x_depth < search_space_depth[0] or int(new_x_depth) == 0:
            continue
        
        ### Sınıflandırı new_x_tree ve new_x_depth parametrelerine göre oluşturulur.
        clf = RandomForestClassifier(n_estimators=int(new_x_tree), max_depth=int(new_x_depth), class_weight='balanced')
        
        ### Sınıflandırıcı eğitilir ve eğitim süresi hesaplanır.
        start_time = time.time()
        clf = clf.fit(train_images, train_labels)
        execution_time = (time.time() - start_time)
        
        ### Sınıflandırıcı val veri seti için tahminde bulunur ve yanılma değeri hesaplanır.
        pred = clf.predict(val_images)
        accuracy = accuracy_score(val_labels, pred)
        loss = 1 - accuracy
        
        ### önce ki durum ile şimdiki durum arasındaki başarı ve eğitim süresi farkı hesaplanır.
        loss_diff = loss - previous_loss
        time_diff = execution_time - execution_time_previous
        
        ### Yeni durumumuz daha uygun bir durum ise durum değişir.
        if loss_diff < 0 or np.random.rand() < math.exp(-loss_diff/T):
            if time_diff < 0 or np.random.rand() < math.exp(-time_diff/T):
                ### x_tree ve x_depth yeni değerlerini alır.
                x_tree = new_x_tree
                x_depth = new_x_depth
                
                ### previous_loss ve execution_time_previous yeni değerlerini alır.
                previous_loss = loss
                execution_time_previous = execution_time
                
                ### Değişimlerin daha sonra grafikte gösterilebilmesi için listelere kaydedilir.
                loss_graph.append(loss)
                time_graph.append(execution_time)
                
        ### Soğukma işlemi gerçekleşir.
        T = T * 0.9
    
    return x_tree, x_depth, loss_graph, time_graph


def find_best_parameter(train_images, train_labels, val_images, val_labels):
    ### Parametre uzayları belirlenir.
    n_estimators = np.linspace(1, 100, num=100)
    max_depth = np.linspace(1, 16, num=16)
    
    ### Simulated Anneling fonksiyonu çağırılır.
    x_tree, x_depth, loss_graph, time_graph = SA(n_estimators, max_depth, train_images, train_labels, val_images, val_labels, T=4)
    
    ### Bulunan değerler geri göndürülür.
    return x_tree, x_depth, loss_graph, time_graph
    

### El yazısı veri seti digits listesine yüklenir.
digits = load_digits()

### Veri seti train, validation, test olarak 3 parçaya ayrılır.
train_images, test_images, train_labels, test_labels = train_test_split(digits.data, digits.target, train_size=0.6, random_state=0)
test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, train_size=0.5, random_state=0)

### Veri setilerinin şekilleri ekrana yazdırılır.
print('train_images: ', train_images.shape)
print('val_images: ', val_images.shape)
print('test_images: ', test_images.shape)

### Eğitim setinin ilk elemanını bastır.
print('train_images[0]: ')
for i in range(0, 8):
    print(train_images[0][i*8:i*8+8])

### Makine öğrenimi modeli için en iyi parametreler Simulated Anneling yöntemi ile aranmaya başlanır.
x_tree, x_depth, loss_graph, time_graph = find_best_parameter(train_images, train_labels, val_images, val_labels)

### Kayıp grafiği çizdirilir.
plt.plot(np.linspace(0, len(loss_graph), num=len(loss_graph)), loss_graph)
plt.title('Loss_graph')
plt.show()

### Modelin eğitim süresi çizdirilir.
plt.plot(np.linspace(0, len(time_graph), num=len(time_graph)), time_graph)
plt.title('Execution_Time_graph')
plt.show()

### İki listeyi aynı grafikte göstermek için ölçeklendirme işlemi yapılır.
time_graph = np.array(time_graph)
loss_graph = np.array(loss_graph)

time_graph = (time_graph - time_graph.min()) / (time_graph.max() + time_graph.min())
loss_graph = (loss_graph - loss_graph.min()) / (loss_graph.max() + loss_graph.min())

### İki grafiğin toplamı tek bir grafikte gösterilir.
plt.plot(np.linspace(0, len(time_graph), num=len(time_graph)), time_graph + loss_graph)
plt.title('Total_Loss_graph')
plt.show()

### Bulunan parametreler ekrana yazırılır.
print('n_estimators: ', int(x_tree))
print('max_depth: ', int(x_depth))

### Sınıflandırı bulunan parametrelere göre oluşturulur.
clf = RandomForestClassifier(n_estimators=int(x_tree), max_depth=int(x_depth), class_weight='balanced')
        
### Sınıflandırıcı eğitilir.
start_time = time.time()
clf = clf.fit(train_images, train_labels)
execution_time = (time.time() - start_time)

### Sınıflandırıcı test veri seti için tahminde bulunur ve yanılma değeri hesaplanır.
pred = clf.predict(test_images)
accuracy = accuracy_score(test_labels, pred)
loss = 1 - accuracy

### Sınıflandırıcı sonuçları ekrana yazdırılır.
print('acc: ', accuracy)
print('loss: ', loss)
print('execution_time: ', execution_time)



