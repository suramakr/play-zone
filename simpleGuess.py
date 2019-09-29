import random
orig = random.randint(1, 20)
# print("Orig number computer has is " + str(orig))
guessed = False
tries = 0

while(not guessed and tries <= 7):
    print("Guess what number I have")
    print("You have " + str(7-tries) + " remaining...")
    try:
        guess = int(input())
    except NameError:
        print("Error in input")
        break
    tries += 1
    if(guess == orig):
        guessed = True
    elif (guess < orig):
        print("number is less")
    elif(guess > orig):
        print("number is greater")

if (guessed == True):
    print("Congrats! you got it in time")
else:
    print("sorry better luck next time")

print("End Program")
