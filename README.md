<h1>Secuencia a secuencia (seq2seq: sequence to sequence)</h1>

https://youtu.be/iynoMdzFmpc?si=Nat_KXNrJErmxXkE

Forward separado por eteapas. (dos secuencias) que se vuelven a pasar por otro forward.

Antes tengo que decir que en este codigo se deja mucho a las librerías. Cosa que evite en las redes anteriores, usando librerías que no me dieran la estructura relativa a la red.

Pero habiendo aprendido MLP, CNN, RNN (LTSM incluido) me puedo permitir usar Torch, (una librería que automatiza varios procesos y nos ofrece estructuras incluidas).

Lo que puede jugar a favor o en contra, depende de como sea el programador. Yo acostumbro a mirar mas el codigo por mi mismo y evitarlas. Al menos en lo que es aprender.

Por lo que, aqui en este resumen, no solo explicaré logica y estructura, sino tambien los puntos que importan a lo que es el uso de la librería.

Por mi parte, jamas había usado Torch. Se me hace que es un arma de doble filo. Oculta demaciado codigo a mi gusto. Y no imagino lo laborioso que puede volverse si llega a darnos algun problema.
Mirando un poco su codigo fuente, a mi gusto, no esta bien documentado.
Pero eso no quita que sin Torch, tendriamos que escribir mucho mas codigo, que en parte ya aprendimos con RNN y LSTM.

De nuevo, como en RNN, es dificil desarrollar un proyecto toy que involucre una tecnología pensada para procesar cantidades inmensas de datos y que refleje buenamente el poder de lo que intentamos aprender.
No es lo que hace, sino como lo hace.

Esta Seq2Seq tiene como proposito predecir una secuencia de numeros aprendida previamente.
De modo que si le dieramamos un numero, aprendiera que secuencia le sigue.

Imaginemos que nuestra red está entrenada con tokens de cifras numericas del estilo

    [[7,2,8,4,3], [6,9,4,7,1], [2,5,1,8,5], ...]

Se le pasa un primer dato a la red para que prediga la secuencia correcta.
A tal dato se lo denomina SOS (Start of Secuence).
Supongamos que el SOS es 7. La red inferirá por lo aprendido que sigue un 2; y si venia un 7 y luego un 2, inferirá que sigue un 8... y asi.

Y si en el batch hubieran dos o más token que comenzaran con 7?

    [[7,2,8,4,3], [7,6,5,2,8], [7,9,1,2,4], ...]

Recordemos que al igual que en RNN, esta es una red pensada para trabajar con secuencias. La salida será la que mas peso tenga, según su aprendizaje. Pero en un uso util real de este tipo de red, podemos usar las salidas con más pesos, para dar opciones y tener un predictor de texto como los que usamos en los celulares.

La estructura de entrenamiento está compuesta de 3 clases e instancian de la siguiente manera.

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

Encoder y Decoder se instancian independientemente. Pero Seq2Seq recibe a los dos como parametros constructores.

Miremos antes la configuración inicial:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10
    seq_length = 5
    embedding_dim = 16
    hidden_dim = 32
    num_epochs = 2000
    batch_size = 64

device funciona como un switch, donde si existe GPU disponible, la usará, sino operará con la CPU.
Nuesto vocabulario, numerico, tiene un tamaño de 10 (numeros del 0 al 9). Y el resto se explican por si mismos.

La red toma el SOS, y lo representa en un vector de pesos randoms.

Miremos la clase Encoder:

    class Encoder(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        def forward(self, x):
            emb = self.embedding(x)
            outputs, (h, c) = self.lstm(emb)
            return h, c

Vemos que en el constructor que se inicia una matriz de valores randoms de tamaño 10x16:

    self.embedding = nn.Embedding(vocab_size, embedding_dim)

No vemos el uso de metodos como random, rand o otros similares. nn.Enbedding es una metodo que ya se ocupa de esto, pues Torch es una librería preparada para este tipo de trabajos.

Esto nos da un array por cada vocabulario de nuestra red. Cada array tiene 16 valores.

    [[ 0.03, -0.21, 0.11, ..., 0.05],  # embedding para el <SOS> 0
    [ 0.12,  0.01, -0.07, ..., 0.08],  # embedding para el <SOS> 1
    ...
    [ 0.09, -0.14, 0.02, ..., -0.03]]  # embedding para el <SOS> 9

Por lo que al pasar al forward el SOS (x) selecciona el embedding para el token 7.

    emb = self.embedding(x)

si nuestro SOS fuera el 1, estaría representado con [ 0.12,  0.01, -0.07, ..., 0.08]

Luego pasa ese embedding por la red LSTM propia de Torch y retorna sus salidas "h" y "c" (ver el repo RNN y LSTM linkeado al final este README).

    outputs, (h, c) = self.lstm(emb)
    return h, c

Aunque aqui no podemos verlo ya que usamos una librería. En LSTM:
h → estado oculto nuevo (salida activada, se usa para predicción). 
c -> el estado interno nuevo (memoria).
También se retorna outputs pero aun no necesitamos ninguna inferencia, por lo tanto se omite su uso. Se retorna solo h y c.

En el Decoder tenemos procesos similares pero con unas diferencias...

Agregamos una transformacion linear Wx + b (Ver MLP), que se inicia con valores al azar.

    self.fc = nn.Linear(hidden_dim, vocab_size)

Ahora aqui, tuve una confusión enorme (por eso no me gustan mucho las librerias, aunque es culpa mia por querer ir rapido) nn.Linear no es una transformación linear vista como una multiplicación de matrices y ya. Es mas que eso. Es una instanciación de una clase Linear y al escribir luego la siguiente linea de codigo:

    logits = self.fc(outputs)

Ejecutamos otro forward que es el verdadero x*W_t + b. Simplificando, multiplicamos los outputs por la transformación linear self.fc. Aqui estaría bueno detenerse a estudiar Torch a fondo. Pero yo no estoy aprendiendo librerias sino redes neuronales. De momento me quedo viendo redes.

    logits = self.fc(outputs)
    es equivalente a:
    logits = torch.matmul(outputs, self.fc.weight.T) + self.fc.bias
    Es decir x*W + b

La transformada de los pesos weight.T y los bias. Son propios de la clase y no de la instancia. Se inician con valores que se iran entrenando.

Decoder finalmente retorna logits, h y c

    return logits, h, c

Se va visualizando ahora el porque del nombre Secuancia a Secuencia. Encoder a Decoder.

Antes de seguir con la explicación, comento un dato interesante que vi en un video. En 2016 Google implementó NMT (Neural Machine Translator) a su traductor. Y fue entonces cuando el traductor de google realmente comenzó a funcionar con la efectividad que conocemos ahora. Antes de eso, no era tan buen traductor.
Originalmente me planteé hacer un traductor como proyecto. Pero asumí que el dataset y entrenamiento podría ser excesivo para mis recursos de hardware (una notebook con GPU limitado de motherboard).

Sin embargo pensemos en el ejemplo del traductor para entender mejor como trabaja seq2seq.

Encoder recibirá una frase en español. Por ejemplo "Hola mundo". Genera las salidas propias de LSTM h y c.
Decorder recibe esta información y produce las salidas. La clase seq2seq es la encargada de gestionar estos cambios entre otras funciones como calcular la perdida.

        Encoder LSTM    ----c--->       Decoder LSTM
    {[Hola] -> [Mundo]} ----h---> {[SOS]:Hello -> [Hello]:World} --h--c--globals-->

Como se observa en la clase Seq2seq, el constructor recibe tanto al encoder como al decoder y los instancia como self.

    self.encoder = encoder
    self.decoder = decoder

En nuestro ciclo for de epocas:

    for epoch in range(num_epochs):
        X, Y = generate_batch(batch_size, seq_length, vocab_size)

Que hace generate_batch():

    def generate_batch(batch_size, seq_length, vocab_size):
        X = np.random.randint(1, vocab_size, (batch_size, seq_length)) # 
        Y = X.copy()  # salida igual a la entrada
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

Declara X como una matriz de numeros randoms enteros que va desde 1 a 9 (1 a vocab_size) y cuyo tamaño es 64x5 (batch_size, seq_length). Copa en Y a X y los retorna como tensores Torch.

tenemos la declaracion de los batchs X e Y. Estos son similares y uno corresponde a la entrada y otro a la salida esperada para el entrenamiento. Concepto que ya vimos antes en otras redes.
Luego pasamos a model los batchs y esto es equivalente a pasarselos al forward.

    output = model(X, Y)
    Es equivalente a:
    output = model.forward(X, Y)
    Porque asi funciona Torch :/

Mirando la clase observamos que forward recibe tales parametros como src y trg:

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            batch_size, trg_len = trg.shape
            vocab_size = self.decoder.fc.out_features
            outputs = torch.zeros(batch_size, trg_len, vocab_size).to(device)

            h, c = self.encoder(src)
            input = trg[:, 0].unsqueeze(1)  # primer token (podría ser un <SOS>)

            for t in range(1, trg_len):
                output, h, c = self.decoder(input, h, c)
                outputs[:, t] = output.squeeze(1)
                top1 = output.argmax(2)
                input = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1

            return outputs

Forward crea una matriz de zeros de tamaño (batch_size, trg_len, vocab_size) llamada outputs.
Luego enviamos el batch que generaremos en el entrenamiento al Forward del encoder, y recibimos h y c.

    h, c = self.encoder(src)

trg (que es el batch generado en el bucle for de el entrenamiento y de forma 64x5) es desmenuzado en batch_size. Es decir en grupos de 64 tokens

    [[3],[3],[9],[2],[4],[2],[9],[5]...[8],[1],[5],[1],[9],[9],[3],[1]]

Eso lo recibe el Decoder tomando el primero como SOS. En el bucle siguiente vemos que junto con los batchs, enviamos h que es la información de como el Encoder memorizo los datos y c que es lo que aprendió a olvidar, a retener y a no aprender (Ver LSTM compuerta ouput).

    for t in range(1, trg_len):
        output, h, c = self.decoder(input, h, c)
        outputs[:, t] = output.squeeze(1)

Como vemos se define output, h y c (para el scoope de la clase seq2seq) que es lo que Decoder procesó y comenzamos a empaquetar los outputs en grupos de a 5. Claro que cada uno esta a su vez agrupado en grupo de 10 valores reales, desde los que se realiza la inferencia. A su vez estos estan agrupados en grupo de 64.

    [[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
    [-0.7767,  0.0528,  0.3603,  ..., -0.0651,  0.1891,  0.0400],
    [-1.0618,  0.0694,  0.3538,  ..., -0.1027,  0.1910,  0.1860],
    [-1.1883,  0.0651,  0.3544,  ..., -0.1331,  0.1952,  0.2317],
    [-1.0589,  0.0467,  0.2717,  ...,  0.0986,  0.2327,  0.2737]],

    [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
    [-0.8705,  0.1724,  0.2353,  ..., -0.0981, -0.0124,  0.2447],
    [-0.9419,  0.2022,  0.2705,  ..., -0.0578,  0.0688,  0.2863],
    [-0.9598,  0.1418,  0.2510,  ...,  0.1393,  0.1948,  0.3417],
    [-0.8650,  0.2022,  0.1934,  ...,  0.1269,  0.1168,  0.2491]],...]x64

Finalmente se calcula la perdida y se actualizan los pesos con el backward automatizado de Torch

    loss = criterion(output[:, 1:].reshape(-1, vocab_size), Y[:, 1:].reshape(-1))
    loss.backward()

Cómo se calcula la perdida y cómo trabajará el backward se predefinen antes del bucle for

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

Creo que no es necesario volver a explicar en este README como funciona un Backward o como se calcula Loss. Aunque existen diferentes maneras de hacerlo, con entender conceptualmente lo que hacen y si es posible, como lo hacen, ya podemos confiarle esa parte a Torch. De todas maneras se pueden ver en los repos anteriores.

Finalmente ordenamos a Torch ejecutar los pasos preseteados

    optimizer.step()

Y ya estara aprendiendo en cada epoch. Vemos que en cada epoch tambien limpia el optimizer

    optimizer.zero_grad()

Realmente no creo que sea necesario seguir explicando el resto del codigo. Son etapas como evaluación y aunque no esta presente, podrian venir otras como el uso directo de la red ya entrenada, o KPIs.

Final.

Creo que si tenemos que mirar el salto que hizo esta red, comparada con RNN, es claro que demuestra la importancia del Forward... Es el corazon de una red.
Aquí tenemos 3 forward LSTM directos (En encoder, en decoder y en el modelo). Y 1 mas, indirecto, si consideramos la transformación linear en el decoder. Pero solo 1 backward.
Eso claramente nos hace ver que los calculos de aprendizaje ocurren en el Forward. Pero en el Backward, los calculos de ajustes sobre esos aprendizajes.
Tambien es llamativo como, aunque el poder de esta red neuronal es mucho mayor a sus antecesoras, en esencia estan ocurriendo los mismos calculos W*x + b. Lo que cambian son las estructuras de como hacemos que sucedan esas transformaciones.
Me imagino, en este punto de mi camino aprendiendo redes neuronales, si volvieramos a obtener una matris cuyos valores fueran calculados nuevamente con otro proceso LSTM. Es decir otra capa mas de forwards trabajando. ¿Que nuevo filtro podriamos aplicar? Probablemente ya se apliquen.

Matrices... Supongo que verlo de una manera geometrica a veces puede ayudar a acostumrar la mente a lo que esta sucediendo tras el codigo. Pero eso una vez que ya se comprendieron los conceptos, sea lo que sea, el dato que las redes representan y estan trabajando, son al fin y al cabo matrices de valores numericos intercambiando datos con operaciones matematicas.
Pero tambien esta bueno ver como el pensamiento puede ser abstraido a representaciones numericas. Creo que es clave separar una cosa de la otra y pensar asi lo que se está programado aqui.


</br>
MLP: https://github.com/Nahuel77/Red_Neuronal_MLP</br>
RNN y LSTM: https://github.com/Nahuel77/Red_Neuronal_RNN_LSTM_incluido</br>