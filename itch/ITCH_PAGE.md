# itch.io page content

Paste these into **Edit game** on itch.io. The images in this folder are sized for itch.io:
`cover.png` is 630x500 (the cover image), and the `screenshot_*.png` files are full-size game screens.

---

## Short description / tagline

Classic backgammon against an AI that looks ahead through every roll of the dice.

## Description (paste into the "Description" box)

**Can you beat an AI that averages over all 21 possible dice rolls before it moves?**

Backgammon vs an Expectiminimax AI is a full game of backgammon you play in your browser against a computer opponent. Pick a difficulty, roll the dice, and try to bear off all 15 checkers first.

### Three levels of opponent

- **Easy**: picks a random legal move. Good for learning the rules.
- **Medium**: greedy. It scores every position it can reach this turn and takes the best one.
- **Hard**: full **Expectiminimax** search. For each of its best candidate moves it considers all 21 dice rolls you might throw next and your best reply to each, then plays the move with the highest expected value.

### See the AI think

After every AI turn, the **AI thinking** panel shows its top three candidate moves with the value the search gave each one, how many moves it searched, and how long it took. While it is searching, the panel shows the MAX, CHANCE and MIN layers it is working through.

Stuck? Press **Hint** and the Hard AI works out the best way to play *your* roll, drawing numbered arrows on the board in the order to play the moves.

### Features

- Complete rules: hitting, the bar, entering, bearing off, doubles played four times, "use both dice if you can" and "use the larger die" rules
- Legal moves highlighted: click a checker, then a green marker
- Undo your moves before the turn ends
- Smooth checker movement, tumbling dice and sound effects
- Pip counts and a move log
- Gammon detection

### Controls

| Action | Control |
|---|---|
| Roll the dice | **Roll** button or **Space** |
| Move a checker | Click the checker, then click a green marker |
| Take back this turn's moves | **Undo** |
| Ask the AI for a hint | **Hint** button or **H** |
| Sound on/off | **Sound** button or **M** |

You play White, moving from point 24 down to point 1. The AI plays Black.

### About

Made by Krishna Mahajan (Student ID 106wucww) for the 5ENT1140 Artificial Intelligence Principles mini project. Written in Python with pygame and published to the web with pygbag.

---

## Suggested page settings

- **Kind of project:** HTML
- **Classification:** Games
- **Genre:** Strategy
- **Tags:** backgammon, board-game, ai, artificial-intelligence, strategy, turn-based, singleplayer, dice, pygame, python
- **Embed options:** Viewport 1180 x 720; tick **Mobile friendly** off; tick **Fullscreen button** on
- **Theme (Edit theme):** background `#161a20`, text `#ebebeb`, link/button colour `#ffc83c`, to match the game

## Screenshots to upload (in this order)

1. `screenshot_hint.png`: the Hint arrows and the Hard AI's suggestion list
2. `screenshot_analysis.png`: the AI thinking panel after an AI turn
3. `screenshot_thinking.png`: the search in progress
4. `screenshot_menu.png`: the title screen
