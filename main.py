import task1, task2

def classify_ionosphere(ionosphere_df):
    X = ionosphere_df[2, 0]
    print(X)
    y = ionosphere_df[0]
    NUM_ITER = 500
    LEARN_RATE = 0.5
    model = task2.LogisticRegression(ionosphere_df, X, y)#, NUM_ITER)
    # theta = model.fit(LEARN_RATE)
    model.fit(LEARN_RATE)
    for i in range(NUM_ITER):
        # theta = model.fit(theta, LEARN_RATE, X, y)
        model.fit(LEARN_RATE)
        if (i % 50) == 0:
            # print(model.cost(X, y, theta))
            print(model.cost())

def main():
    df_tuple = task1.clean_data(False)
    ionosphere = df_tuple[0]
    adult = df_tuple[1]
    ttt = df_tuple[2]

    classify_ionosphere(ionosphere)

if __name__ == "__main__":
    main()
