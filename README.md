# sdlarch-rl

This is a fork of sdlarch that aims to develop RL (Reinforcement Learning) projects.

## sdlarch

sdlarch is a small libretro frontend (sdlarch.c has less than 1000 lines of
code) created for educational purposes. It only provides the required (video,
audio and basic input) features to run basic libretro cores and there's no UI
or configuration support.

## Building

### Linux:

```shell
cmake .
make
```

### Windows:

```shell
cmake .
nmake
```

## TODO

- [ ] Run Dolphin Core
- [x] Run PCSX2 Core
- [x] Load PCSX2 States
- [ ] Compile cores in build
- [ ] Tool to add games/map memory/save states, etc.
- [x] Load PCSX2 state from file
- [ ] Gymnasium compatibility
- [x] Load Emulator memory
- [ ] Load games in the same standard as stable-retro
- [ ] Improve performance
- [ ] Load another cores (Dolphin, etc)

## Our Youtube Channel

If you are interested in our AI projects, visit our channel:

[AI Brain](https://www.youtube.com/@AiBrainAi?sub_confirmation=1)


