# How the plate reader works

*GodsEye · Team Prometheus · written for a non-technical reader*

---

## The problem

A traffic camera sits on a pole, five to eight metres up, looking down at cars going past.
It takes a picture. Somewhere in that picture is a number plate — maybe 100 pixels wide,
about the size of a postage stamp on your screen.

It might be raining. It might be night. The plate might be covered in road dirt. The car
might be moving fast enough to blur.

Our job: turn that into letters and numbers, reliably enough that a police officer can act
on it.

---

## Step 1 — Find the plate

Before you can read a plate you have to find it in the picture.

A number plate has a giveaway signature: it is a **wide rectangle full of closely-spaced
vertical strokes**. Letters and digits are mostly up-and-down lines. Nothing else on a car
looks quite like that — a bumper is smooth, a windscreen is smooth, a grille has lines but
they run the wrong way.

So the software looks for busy patches, smears them together so the individual letters
merge into one blob, and keeps only the blobs that are the right shape — roughly two to
seven times wider than tall.

**This step works about 57% of the time**, and it is the weakest part of the whole system.
More on that at the end.

---

## Step 2 — Reading the letters: the obvious way, and why it fails

The obvious approach is what a human would do:

1. Cut the plate into individual characters
2. Look at each character and decide what it is
3. Stick the answers together

We built this first. **It scores 16% on realistic camera images.**

Here is why. Look at a badly photographed plate and try to say exactly where one character
ends and the next begins. On a clean plate it is obvious. On a plate that has been through
a cheap camera, compressed, and blown up again, the gaps between characters simply *are not
in the picture any more*.

> **Cutting first is a decision made too early.** You are forced to commit to where the
> letters start and stop before you have read anything — and if you cut in the wrong place,
> everything after it is wrong. There is no recovering.

It is like being handed a smudged handwritten word and told: first draw lines between the
letters, *then* read it. If you draw the lines wrong you will read it wrong, and you had no
way of knowing.

---

## Step 3 — The way that works: read it all at once

The system we actually use never cuts the plate up.

Instead it slides across the plate from left to right, and at each position asks: *at this
exact spot, how much does this look like an A? a B? a 7?* — producing a running commentary
of opinions across the whole width.

Then a piece of mathematics does something clever. It considers **every possible way** those
opinions could be grouped into letters, and picks the grouping that makes the most sense
overall. It never has to decide where the letters begin. The boundaries fall out of the
answer rather than being fixed in advance.

**Same images, same difficulty: 16% → 43%.** Nearly three times better, and about four
times faster.

There is a second benefit. The reader has a memory that runs along the plate, so it learns
the *shape* of an Indian registration — two letters, then digits, then letters, then four
digits. When character four is a smudge, it already knows a digit belongs there. We also
enforce that rule explicitly: a letter can never be placed in a slot where only a digit is
allowed.

---

## Step 4 — Knowing when you do not know

This is the part worth understanding above all the others, because it is a design
philosophy rather than a trick.

**The system does not have to answer.** When it is not confident, it throws the reading
away rather than guessing.

The reason is simple. A *wrong* number plate handed to a police officer is far worse than
*no* number plate. A blank means "we did not get that one." A wrong answer means an
innocent motorist receives a notice for something they did not do.

So every reading gets a confidence score, and anything below the bar is discarded. That is
why there are two numbers, and why they are different:

| | |
|---|---|
| **73%** | of all plates, how many it reads correctly |
| **96%** | of the readings it is confident enough to keep, how many are exactly right |

It keeps roughly four readings in five and refuses the rest.

**A discovery worth mentioning.** Our first confidence score was broken in a way nobody
noticed. It was *worse than a coin flip* — systematically more confident about wrong
answers than right ones. We replaced it with a proper calculation, and at the same accuracy
level it now keeps **three times as many readings**.

The lesson: a number that looks reasonable can still be backwards, and you only find out by
measuring it.

---

## Step 5 — Take several photographs, not one

This is the single biggest improvement in the whole project, and it required no cleverness
at all.

A real traffic camera is not a photographer taking one shot. It is triggered as the car
enters the zone and takes a **burst** — five, ten, fifteen frames as the vehicle crosses.
Different distances, different blur, different glare.

Most systems read one of them. We read all of them and let them vote.

The reason it works so well is that the frames fail in *different* ways. A frame that loses
the plate to a headlight reflection is outvoted by the four that did not. It is the same
reason you take several photographs of a bird rather than trusting one, or why a laboratory
repeats a measurement.

| frames read | plates read correctly |
|---|---|
| 1 | 42% |
| 5 | 73% |
| 8 | 77% |

Same software, same camera, no retraining.

---

## Step 6 — Ask the neighbouring cameras

When a camera fails completely, we do not quite give up.

A few minutes later the cameras up and down the road have reported in. Whatever drove past
the failed camera almost certainly appears in **one of those lists**. So instead of guessing
from the 13.6 trillion possible registrations, we take the few hundred plates the neighbours
actually saw and ask: which of these best explains the blurry image we could not read?

It is the difference between *"name this person"* and *"which of these forty people is
it?"* — a far easier question.

**On our test day it recovered 14 of 25 unreadable captures, and all 14 were right.** Not
near-misses either: one went from `DL3HMV8199` to `OD39HMV8199` — a different state and
four different characters.

Two rules keep it honest:

- A recovered plate is filed **separately** and labelled a *guess*, never mixed in with
  plates that were genuinely read.
- A guess can never become the evidence for another guess. Otherwise one mistake spreads
  down a corridor and the system invents a journey nobody made.

---

## What is still wrong

We know where the remaining problem is, because we tested for it rather than assuming.

**The information is there. Our reader is not finding it.** We checked whether the plate is
even recoverable from the difficult images, and in most cases it is. The pictures are good
enough; the reader is not good enough yet. Only in the "storm" case — everything wrong at
once — is the plate genuinely destroyed, and refusing those is the correct behaviour.

**And we found the likely reason by accident.** We showed the reader a *perfect, clean*
number plate: no rain, no blur, no compression.

It scored **4%**.

It gets *better* as you make the picture worse.

That is the tell. Every image we trained it on was heavily degraded, so it has never seen a
sharp plate. It has learned the fingerprint of our particular camera — its blur, its
compression artefacts — rather than the shapes of the letters themselves. Like a student who
memorised the smudges on one photocopy and then cannot read the clean original.

Fixing that means showing it easy examples alongside the hard ones. That is the next
experiment.

---

## One honest caveat about the 73%

That figure is measured on plates that have **already been found** in the picture.

Feed the system a whole photograph and the finding step succeeds 57% of the time — so end
to end, from a full camera frame, it is closer to **22%**.

The reader is the strong part. Finding the plate is the weak part, and it is still a
hand-written rule of thumb rather than something trained. Replacing it is the
highest-value fix we know of.

---

## In one page

1. **Find** the plate by looking for a wide patch of vertical strokes.
2. **Read** it whole rather than cutting it into letters first — cutting is a decision made
   too early, and it is what most systems get wrong.
3. **Refuse** to answer when unsure, because a wrong plate is worse than no plate.
4. **Vote** across the several frames a camera actually takes, because they fail in
   different ways.
5. **Ask the neighbours** when a camera fails outright, turning an impossible guess into a
   multiple-choice question.
6. **Be honest** about the fact that the reader has learned our camera's flaws rather than
   the alphabet, and that finding the plate — not reading it — is now the bottleneck.
