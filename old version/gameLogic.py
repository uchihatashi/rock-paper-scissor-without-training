import random
import cv2

class rockPaperScissorsGame:

    def __init__(self):
        self.choice = 0
        self.choices = ["rock", "paper", "scissor"]
        self.gesture = ""

    def makeChoice(self):
        for i in range(random.randint(0, 3)):
            self.choice = i
        
        self.gesture = self.choices[self.choice]

        
    def printChoice(self):
        print (self.gesture)


def runGame():
    game = rockPaperScissorsGame()
    game.makeChoice()
    return game.gesture

super_logic={
    'rock_scissor':'you won',
    'rock_paper':'you lost',
    'rock_rock':'its a draw',
    'paper_rock':'you won',
    'paper_scissor':'you lost',
    'paper_paper':'its a draw',
    'scissor_paper':'you won',
    'scissor_rock':'you lost',
    'scissor_scissor':'its a draw',
}
showme={
    'rock' : cv2.imread('./assets/rock.png'),
    'paper' : cv2.imread('./assets/paper.png'),
    'scissor' : cv2.imread('./assets/scissor.png')
}

def logic(human,computer):
    game_key= human + "_" + computer
    
    return super_logic[game_key],showme[computer]



