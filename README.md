# sdlarch-rl

This is a fork of sdlarch that aims to develop RL (Reinforcement Learning) projects.

## sdlarch

sdlarch is a small libretro frontend (sdlarch.c has less than 1000 lines of
code) created for educational purposes. It only provides the required (video,
audio and basic input) features to run basic libretro cores and there's no UI
or configuration support.

## Building

- Linux:

```shell
cmake .
make
```

- Windows:

```shell
cmake .
nmake
```

## TODO

- [x] Run PCSX2 Core
- [ ] Load PCSX2 States
- [ ] Gymnasium compatibility
- [ ] Load Emulator memory
- [ ] Load games in the same standard as stable-retro
- [ ] Improve performance
- [ ] Load another cores (Dolphin, etc)

## Our Youtube Channel

If you are interested in our AI projects, visit our channel:

[AI Brain](https://www.youtube.com/@AiBrainAi?sub_confirmation=1)
