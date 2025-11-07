<h1>Secuencia a secuencia (seq2seq: sequence to sequence)</h1>

https://youtu.be/iynoMdzFmpc?si=Nat_KXNrJErmxXkE

Forward separado por eteapas. (dos secuencias) que se vuelven a pasar por otro forward.

Antes tengo que decir que en este codigo se deja mucho a las librerías. Cosa que evite en las redes anteriores, usando librerías que no me dieran la estructura relativa a la red.

Pero habiendo aprendido MLP, CNN, RNN (LTSM incluido) me puedo permitir usar Torch, (una librería que automatiza varios procesos y nos ofrece estructuras incluidas).

Lo que puede jugar a favor o en contra, depende de como sea el programador. Yo por mi parte acostumbro a mirar mas el codigo por mi mismo y evitarlas. Al menos en lo que es aprender.

Por lo que, aqui en este resumen, no solo explicaré logica y estructura, sino tambien los puntos que importan a lo que es el uso de la librería.

Por mi parte, jamas había usado Torch. Se me hace que es un arma de doble filo. Oculta demaciado codigo a mi gusto. Y no imagino lo laborioso que puede volverse si llega a darnos algun problema.
Mirando un poco su codigo fuente, a mi gusto, no esta bien documentado.
Pero eso no quita que sin Torch, tendriamos que escribir mucho mas codigo, que en parte ya aprendimos con RNN y LSTM.

De nuevo, como en RNN, es dificil desarrollar un proyecto toy, que involucra una tecnología pensada para procesar cantidades inmensas de datos, que refleje buenamente el poder de lo que intentamos aprender.
No es lo que hace, sino como lo hace.

Esta Seq2Seq tiene como proposito predecir una secuencia de numeros aprendida previamente.
De modo que si le dieramamos un numero, aprendiera que secuencia le sigue.

Imaginemos que nuestra red esta entrenada con tokens de cifras numericas del estilo

    [[7,2,8,4,3], [6,9,4,7,1], [2,5,1,8,5], ...]

Se le pasa un primer dato a la red para que prediga la secuencia correcta.
A tal dato se lo denomina SOS.
Supongamos que el SOS es 7. La red inferirá por lo aprendido que sigue un 2; y si venia un 7 y luego un 2, inferirá que sigue un 8... y asi.

Y si en el batch hubieran dos o más token que comenzaran con 7?

    [[7,2,8,4,3], [7,6,5,2,8], [7,9,1,2,4], ...]

Recordemos que al igual que en RNN, esta es una red pensada para trabajar con secuencias. La sera la que mas peso tenga, según su aprendizaje, pero en un uso util real de este tipo de red, podemos usar las salidas con más pesos, para dar opciones y tener un predictor de texto como los que usamos en los celulares.

La estructura de entrenamiento esta compuesta de 3 clases y instancia de la siguiente manera.

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

Vemos que en el constructor se inicia una matriz de valores randoms de tamaño 10x16:

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

Aunque aqui no podemos verlo ya que usamos una librería. En LSTM h → estado oculto nuevo (salida activada, se usa para predicción), c el estado interno nuevo (memoria). También se retorna outputs pero aun no necesitamos ninguna inferencia, por lo tanto se omite su uso. Se retorna solo h y c.

En el Decoder tenemos procesos similares pero con unas diferencias...

Agregamos una transformacion linear Wx + b (Ver MLP), que se inicia con valores al azar.

    self.fc = nn.Linear(hidden_dim, vocab_size)

Ahora aqui, tuve una confusión enorme (por eso no me gustan mucho las librerias, aunque es culpa mia por querer ir rapido) nn.Linear no es una transformación linear vista como una multiplicación de matrices y ya. Es mas que eso. Es una instanciación de una clase y al escribir luego la siguiente linea de codigo:

    logits = self.fc(outputs)

Ejecutamos otro forward que es el verdadero xW_t + b. Simplificando, multiplicamos los outputs por la transformación linear self.fc. Aqui estaría bueno detenerse a estudiar Torch a fondo. Pero yo no estoy aprendiendo librerias sino redes neuronales.

    logits = self.fc(outputs)
    es equivalente a:
    logits = torch.matmul(outputs, self.fc.weight.T) + self.fc.bias

La transformada de los pesos weight.T y los bias. Son propios de la clase y no de la instancia. Se inician con valores que se iran entrenando.

Decoder finalmente retorna logits, h y c

    return logits, h, c



</br>
MLP: https://github.com/Nahuel77/Red_Neuronal_MLP</br>
RNN y LSTM: https://github.com/Nahuel77/Red_Neuronal_RNN_LSTM_incluido</br>