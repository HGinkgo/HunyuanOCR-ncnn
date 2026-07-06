# Example Output Excerpts

These are short excerpts from the PyTorch fp32 reference outputs used by the
current `max_pixels=524288` validation. They are model reference outputs, not
manually corrected OCR ground truth. The C++/ncnn regression compares the full
generated token sequence and decoded text through fixture files; this document
is only a lightweight preview for the bundled example images.

| Case | Prompt mode | Grid | New tokens | Output chars |
| --- | --- | --- | ---: | ---: |
| `hf_demo` | `spotting` | `38x52` | 411 | 1150 |
| `chinese_doc` | `spotting` | `54x36` | 593 | 859 |
| `en_book` | `document` | `58x34` | 1024 | 3437 |
| `formula` | `document` | `58x34` | 979 | 2961 |
| `table` | `document` | `54x36` | 325 | 1770 |

## hf_demo

```text
Lab mouse portrait(142,15),(251,33)huggingface.co/chat/conversation/6655bacb6dfed1a136...(156,76),(545,99)R(644,78),(654,93)Relaunch to update...
Generate the portrait of a scientific mouse in its laboratory.(151,176),(557,199)Called tool Image Generation...
```

## chinese_doc

```text
7.脾虚呕吐(126,106),(246,123)【表现】(171,137),(242,154)脾胃虚弱，神疲乏力。
【家庭复方调理法】(171,202),(353,219)推拿法：每天做一次。
```

## en_book

```text
Gotu kola, kava kava, St. John's wort, valerian may increase CNS depression. FOOD: None known.
LAB VALUES: May decrease total free thyroxine ($ T_{1} $) serum levels...
```

## formula

```text
Let $ g,h $ be the inverse Fourier transforms of $ \chi_{U},\chi_{UK} $ (as given by the Plancherel theorem), and let $ f=|U|^{-1}gh $...
(4.51) Theorem. If $ N\subset\widehat{G} $ is closed, then $ \nu(\iota(N))=N $.
```

## table

```text
Fig. 10.3 Solved sudoku puzzle.

## 10.4 Hybrid Optimization

Hybrid methods may be required to solve particularly difficult real-world optimization problems...
```
