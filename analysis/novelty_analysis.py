#%%
import pandas as pd
import transformers
import torch
import numpy as np
#%%
# model_path = "Qwen/Qwen3-0.6B"
model_path = "Qwen/Qwen3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
#%%
# load data/dapo-math-17k.parquet
dataset = pd.read_parquet("data/dapo-math-17k.parquet")
#%%
question = dataset["prompt"][0][0]["content"]
ground_truth = dataset["reward_model"][0]["ground_truth"]
ground_truth_solution = r"""
Let (AB=x) and (AC=y). Place coordinates so that

* (A=(0,0)),
* (B=(x,0)),
* (\angle A=\angle BAC) is acute with (\sin A=\frac45), so (\cos A=\frac35),
* hence (C=(y\cos A,;y\sin A)=\left(\frac{3y}{5},\frac{4y}{5}\right)).

Since (\angle BAD=\angle DAC), the ray (AD) is the internal angle bisector of (\angle A). Also (AD=1), so (D) is the point at distance (1) from (A) on that bisector.

Compute half-angle values:
[
\cos\frac A2=\sqrt{\frac{1+\cos A}{2}}=\sqrt{\frac{1+\frac35}{2}}=\sqrt{\frac45}=\frac{2}{\sqrt5},\qquad
\sin\frac A2=\sqrt{\frac{1-\cos A}{2}}=\sqrt{\frac{1-\frac35}{2}}=\sqrt{\frac15}=\frac{1}{\sqrt5}.
]
Thus
[
D=\left(\cos\frac A2,\sin\frac A2\right)=\left(\frac{2}{\sqrt5},\frac{1}{\sqrt5}\right).
]
Let (s=\frac{1}{\sqrt5}). Then (D=(2s,s)).

---

## Use (\angle BDC=90^\circ)

(\angle BDC=90^\circ) means ((\overrightarrow{DB})\cdot(\overrightarrow{DC})=0).

[
\overrightarrow{DB}=B-D=(x-2s,,-s),\qquad
\overrightarrow{DC}=C-D=\left(\frac{3y}{5}-2s,;\frac{4y}{5}-s\right).
]
Dot product:
[
(x-2s)\left(\frac{3y}{5}-2s\right)+(-s)\left(\frac{4y}{5}-s\right)=0.
]
Expand:
[
\frac{3xy}{5}-2sx-2sy+5s^2=0.
]
Since (s^2=\frac15), we have (5s^2=1), so
[
\frac{3xy}{5}-2s(x+y)+1=0. \tag{1}
]

---

## Use (\dfrac{BD}{CD}=\dfrac{3}{2})

Compute squared distances:
[
BD^2=(x-2s)^2+s^2=x^2-4sx+5s^2=x^2-4sx+1,
]
[
CD^2=\left(\frac{3y}{5}-2s\right)^2+\left(\frac{4y}{5}-s\right)^2
= y^2-4sy+5s^2=y^2-4sy+1.
]
Thus
[
\frac{BD^2}{CD^2}=\frac{x^2-4sx+1}{y^2-4sy+1}=\frac{9}{4}. \tag{2}
]

---

## Clear radicals by scaling

Let
[
X=\frac{x}{s}=x\sqrt5,\qquad Y=\frac{y}{s}=y\sqrt5.
]
Divide (1) by (s^2) (since (s\neq 0)) and use (xy/s^2=XY), ((x+y)/s=X+Y), and (1/s^2=5):
[
\frac{3}{5}XY-2(X+Y)+5=0 ;\Longrightarrow; 3XY-10(X+Y)+25=0. \tag{3}
]
Also divide numerator/denominator in (2) by (s^2):
[
\frac{X^2-4X+5}{Y^2-4Y+5}=\frac94. \tag{4}
]

From (3):
[
X(3Y-10)=10Y-25 ;\Longrightarrow; X=\frac{10Y-25}{3Y-10}. \tag{5}
]
Substitute into (4). This simplifies to the factorization
[
(9Y-40)(9Y-20)(Y^2-4Y+5)=0,
]
so the real solutions are
[
Y=\frac{20}{9}\quad\text{or}\quad Y=\frac{40}{9}.
]
Using (5):

* If (Y=\frac{20}{9}), then (X=\frac{5}{6}).
* If (Y=\frac{40}{9}), then (X=\frac{35}{6}).

---

## Use “(D) is outside (\triangle ABC)” to choose the correct solution

Since (D) lies on the internal angle bisector from (A) and (AD=1), (D) is outside (\triangle ABC) exactly when the angle-bisector segment (AE) (where (E=AD\cap BC)) satisfies (AE<1).

The angle-bisector length is
[
AE=\frac{2xy\cos\frac A2}{x+y}.
]
We know (\cos\frac A2=\frac{2}{\sqrt5}=2s).

### Case 1: ((X,Y)=\left(\frac56,\frac{20}{9}\right))

Then (x=sX,\ y=sY), so
[
x+y=s(X+Y)=s\left(\frac56+\frac{20}{9}\right)=s\cdot\frac{55}{18},
]
[
xy=s^2XY=\frac15\cdot\frac56\cdot\frac{20}{9}=\frac{10}{27}.
]
Thus
[
AE=\frac{2\cdot \frac{10}{27}\cdot (2s)}{s\cdot\frac{55}{18}}
=\frac{\frac{40}{27}s}{\frac{55}{18}s}
=\frac{40\cdot 18}{27\cdot 55}
=\frac{16}{33}<1,
]
so (D) is indeed outside. This is the valid case.

### Case 2: ((X,Y)=\left(\frac{35}{6},\frac{40}{9}\right))

A similar computation gives (AE=\frac{224}{111}>1), so (D) would lie inside (\triangle ABC), invalid.

Hence we use Case 1.

---

## Compute (AB+AC)

[
AB+AC=x+y=s(X+Y)=\frac{1}{\sqrt5}\cdot\frac{55}{18}
=\frac{55}{18\sqrt5}
=\frac{11\sqrt5}{18}.
]
So (a=11,\ b=5,\ c=18), pairwise coprime, and
[
a+b+c=11+5+18=34.
]

Answer: 34
"""

random_solution = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lobortis ut orci quis imperdiet. Donec justo nibh, gravida vitae libero ac, congue tincidunt velit. In consectetur elit et ipsum tristique, quis ullamcorper mauris auctor. Cras sed lectus porta, pretium purus in, fermentum quam. Vivamus ligula urna, dapibus at rhoncus ut, cursus id elit. Nam nec convallis risus. Duis eu accumsan neque. Suspendisse sit amet augue purus.

Integer vulputate sem vel molestie maximus. Vivamus pharetra mauris id quam dapibus bibendum. Quisque fermentum est vitae arcu vulputate posuere. Fusce urna dolor, commodo ut sem vitae, blandit sagittis dolor. Proin mauris justo, tincidunt in rhoncus at, tristique eu diam. Sed eleifend et ante vitae viverra. Donec hendrerit facilisis eros, ac aliquam lectus cursus nec. Sed eget venenatis mi. Ut ullamcorper blandit ipsum, in auctor diam mollis eu. Morbi pretium quam vitae arcu hendrerit, et rhoncus nunc venenatis. Vivamus semper et nisl ut eleifend. Aenean non fringilla mi, eu convallis tellus."""

another_questions_solution = r"""
Looking at this problem, I'll set up coordinates with $A = (0,0)$, $B = (1,0)$, $C = (1,1)$, $D = (0,1)$.

## Parameterization

I'll parameterize the perimeter by arc length $s \in [0,4)$ starting from $A$:
- $s \in [0,1)$: side $AB$ → point $(s, 0)$
- $s \in [1,2)$: side $BC$ → point $(1, s-1)$  
- $s \in [2,3)$: side $CD$ → point $(3-s, 1)$
- $s \in [3,4)$: side $DA$ → point $(0, 4-s)$

For triangle $AXY$ with $A = (0,0)$, $X = (x_1, y_1)$, $Y = (x_2, y_2)$:
$$\text{Area} = \frac{1}{2}|x_1 y_2 - x_2 y_1|$$

The expected area is:
$$E[\text{Area}] = \frac{1}{16} \int_0^4 \int_0^4 \frac{1}{2}|x_1(s) y_2(t) - x_2(t) y_1(s)| \, ds \, dt$$

## Case Analysis

I'll compute the contribution from each pair of sides. Let $f(s,t) = x_1(s)y_2(t) - x_2(t)y_1(s)$.

**Both points on same side (collinear with $A$):**
- Cases: $(AB, AB)$ and $(DA, DA)$ → Contribution: $0$

**Adjacent sides $(AB, BC)$ and $(BC, AB)$:**
- $X = (s,0)$, $Y = (1, t-1)$: $f = s(t-1) \geq 0$
- $\int_0^1 \int_1^2 \frac{s(t-1)}{2} = \frac{1}{2} \cdot \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{8}$
- By symmetry, both cases give $\frac{1}{8}$ each → Total: $\frac{1}{4}$

**Opposite sides $(AB, CD)$ and $(CD, AB)$:**
- $X = (s,0)$, $Y = (3-t,1)$: $f = s \geq 0$  
- $\int_0^1 \int_2^3 \frac{s}{2} = \frac{1}{4}$
- Both cases give $\frac{1}{4}$ each → Total: $\frac{1}{2}$

**Adjacent sides $(AB, DA)$ and $(DA, AB)$:**
- Similar to $(AB,BC)$ → Total: $\frac{1}{4}$

**Both points on $BC$:**
- $X = (1, s-1)$, $Y = (1, t-1)$: $f = t-s$
- $\int_1^2 \int_1^2 \frac{|t-s|}{2} = \frac{1}{2} \cdot \frac{1}{3} = \frac{1}{6}$

**Adjacent sides $(BC, CD)$ and $(CD, BC)$:**
- $X = (1, s-1)$, $Y = (3-t, 1)$: $f = 1 - (3-t)(s-1)$
- With $u = s-1 \in [0,1)$, $v = 3-t \in (0,1]$:
- $\int_0^1 \int_0^1 \frac{1-uv}{2} dv\,du = \frac{1}{2}(1 - \frac{1}{4}) = \frac{3}{8}$
- Both cases give $\frac{3}{8}$ each → Total: $\frac{3}{4}$

**Opposite sides $(BC, DA)$ and $(DA, BC)$:**
- Each contributes $\frac{1}{4}$ → Total: $\frac{1}{2}$

**Both points on $CD$:**  
- By symmetry with $BC$ case → Total: $\frac{1}{6}$

**Adjacent sides $(CD, DA)$ and $(DA, CD)$:**
- Each contributes $\frac{1}{8}$ → Total: $\frac{1}{4}$

## Final Calculation

Sum of all contributions:
$$0 + \frac{1}{4} + \frac{1}{2} + \frac{1}{4} + \frac{1}{6} + \frac{3}{4} + \frac{1}{2} + \frac{1}{6} + \frac{1}{4} = \frac{17}{6}$$

Expected area:
$$E[\text{Area}] = \frac{1}{16} \cdot \frac{17}{6} = \frac{17}{96}$$

Since $\gcd(17, 96) = 1$, we have $m = 17$ and $n = 96$.

$$m + n = \boxed{113}$$ """


new_solution = r"""
put the **right angle point** (D) at the origin and use **(i)** dot/cross to encode (\tan A), and **(ii)** the “point on angle bisector (\Leftrightarrow) equal distances to the two sides” characterization, with distances computed by **area determinants**.

---

## 1) Coordinate setup centered at (D)

Given (\angle BDC=90^\circ) and (\dfrac{BD}{CD}=\dfrac{3}{2}), scale so
[
D=(0,0),\quad B=(3t,0),\quad C=(0,2t)\qquad (t>0).
]
Also (AD=1), so (A) lies on the unit circle about the origin:
[
A=(p,q),\quad p^2+q^2=1.
]
Let
[
AB=|B-A|,\quad AC=|C-A|.
]

---

## 2) Encode (\angle A) using dot/cross at (A)

We know (\sin A=\frac45) and (A<90^\circ), hence (\cos A=\frac35) and
[
\tan A=\frac{\sin A}{\cos A}=\frac{4}{3}.
]

Vectors:
[
\overrightarrow{AB}=B-A=(3t-p,,-q),\quad \overrightarrow{AC}=C-A=(-p,,2t-q).
]

Compute dot and (2D) cross determinant:
[
\overrightarrow{AB}\cdot\overrightarrow{AC}=(3t-p)(-p)+(-q)(2t-q)=1-t(3p+2q),
]
[
\overrightarrow{AB}\times\overrightarrow{AC}=\det(\overrightarrow{AB},\overrightarrow{AC})
=(3t-p)(2t-q)-(-q)(-p)=6t^2-t(2p+3q).
]

Since (\tan A=\dfrac{|\overrightarrow{AB}\times\overrightarrow{AC}|}{\overrightarrow{AB}\cdot\overrightarrow{AC}}) and (\overrightarrow{AB}\cdot\overrightarrow{AC}>0) (acute angle), we have two sign possibilities:
[
\frac{6t^2-t(2p+3q)}{1-t(3p+2q)}=\pm\frac{4}{3}.
]
The branch that eventually matches “(D) is outside (\triangle ABC)” is the **minus** branch; multiplying out gives:
[
3(6t^2-t(2p+3q))=-4(1-t(3p+2q)).
]
Rearrange:
[
18t^2-(18p+17q)t+4=0. \tag{★}
]

---

## 3) Use the angle bisector condition via equal distances

(\angle BAD=\angle DAC) means (D) lies on the angle bisector of the two lines (AB) and (AC).
Equivalently, the (unsigned) distances from (D) to the lines (AB) and (AC) are equal:
[
\operatorname{dist}(D,AB)=\operatorname{dist}(D,AC).
]

Distance from origin to a line through (A) and (B) can be written as
[
\operatorname{dist}(0,AB)=\frac{|[A,B]|}{|AB|},
]
where ([A,B]=\det(A,B)) is twice the signed area of (\triangle 0AB).

Compute the determinants:
[
[A,B]=\det((p,q),(3t,0))=-3tq,\quad |[A,B]|=3t|q|,
]
[
[A,C]=\det((p,q),(0,2t))=2tp,\quad |[A,C]|=2t|p|.
]
So
[
\frac{3t|q|}{AB}=\frac{2t|p|}{AC}\quad\Longrightarrow\quad \frac{AB}{AC}=\frac{3|q|}{2|p|}.
]
Let (u=\frac{|q|}{|p|}\ge 0). Then
[
\frac{AB}{AC}=\frac{3u}{2}. \tag{1}
]

Now introduce (w=t|p|) (so (t=\dfrac{w}{|p|})). Since (p^2+q^2=1), we have (|p|=\dfrac{1}{\sqrt{1+u^2}}).

Compute squared lengths (using (p^2+q^2=1)):
[
AB^2=(3t-p)^2+q^2=9t^2-6tp+1,
]
[
AC^2=p^2+(2t-q)^2=4t^2-4tq+1.
]
In terms of (u,w) (and (|p|=\frac{1}{\sqrt{1+u^2}})) this becomes
[
AB^2=9w^2(1+u^2)-6w+1,\qquad
AC^2=4w^2(1+u^2)-4uw+1.
]
Square equation (1): (\dfrac{AB^2}{AC^2}=\dfrac{9u^2}{4}). Cross-multiplying and simplifying *factors linearly* in (w), giving
[
6w(u^2-1)=3u-2. \tag{2}
]

---

## 4) Eliminate (t,p,q): solve for (u)

From (2),
[
w=\frac{3u-2}{6(u^2-1)}.
]

Now rewrite the angle equation (★) in the same variables. With the chosen sign branch, it becomes (after substituting (t=\frac{w}{|p|}) and (|p|=\frac{1}{\sqrt{1+u^2}})) the quadratic:
[
18w^2(1+u^2)-w(18+17u)+4=0. \tag{3}
]

Plug (2)’s (w) into (3). The resulting equation in (u) collapses nicely to
[
(4u-1)(7u-8)=0,
]
so
[
u=\frac14 \quad\text{or}\quad u=\frac87.
]

---

## 5) Use “(D) outside (\triangle ABC)” to pick the correct root

Let (E=AD\cap BC). If (D) is **outside** (\triangle ABC) on the angle bisector ray, then (AD>AE) (i.e., (D) lies beyond (BC)).

The angle-bisector length from (A) to (BC) is
[
AE=\frac{2\cdot AB\cdot AC\cdot \cos\frac{A}{2}}{AB+AC}.
]
Here (\cos\frac{A}{2}=\sqrt{\frac{1+\cos A}{2}}=\sqrt{\frac{1+\frac35}{2}}=\frac{2}{\sqrt5}).

* For (u=\frac14), solving (2) gives (w=\frac29), hence
  [
  AB^2=\frac{5}{36}\Rightarrow AB=\frac{\sqrt5}{6},\qquad
  AC^2=\frac{80}{81}\Rightarrow AC=\frac{4\sqrt5}{9}.
  ]
  Then
  [
  AE=\frac{2\cdot\frac{\sqrt5}{6}\cdot\frac{4\sqrt5}{9}\cdot\frac{2}{\sqrt5}}{\frac{\sqrt5}{6}+\frac{4\sqrt5}{9}}
  =\frac{16}{33}<1=AD,
  ]
  so (D) is outside (good).

* For (u=\frac87), you get (AE>1), meaning (D) lies between (A) and (BC) (inside), which violates the condition.

So we take (u=\frac14).

---

## 6) Finish: compute (AB+AC)

From the valid case:
[
AB+AC=\frac{\sqrt5}{6}+\frac{4\sqrt5}{9}
=\sqrt5\left(\frac{3}{18}+\frac{8}{18}\right)
=\frac{11\sqrt5}{18}.
]
So (\frac{a\sqrt b}{c}=\frac{11\sqrt5}{18}) with ((a,b,c)=(11,5,18)), hence
[
a+b+c=11+5+18=34.
]

Answer: 34
"""

ground_truth_solution_version_2 = r"""


Let (AB=x,\ AC=y). Put (A) at the origin and put (AB) on the (x)-axis:

[
A=(0,0),\quad B=(x,0).
]

Given (\sin A=\frac45) and (A) acute, (\cos A=\frac35). Hence

[
C=\big(y\cos A,\ y\sin A\big)=\left(\frac{3y}{5},\frac{4y}{5}\right).
]

### 1) Locate (D) from the angle-bisector and (AD=1)

(\angle BAD=\angle DAC) means (AD) is the internal bisector of (\angle A). Thus the direction of (AD) makes angle (A/2) from (AB). Since (AD=1),

[
D=(\cos \tfrac A2,\ \sin \tfrac A2).
]

Compute half-angle values using (\cos A=\frac35):

[
\cos \frac A2=\sqrt{\frac{1+\cos A}{2}}=\sqrt{\frac{1+\frac35}{2}}=\sqrt{\frac45}=\frac{2}{\sqrt5},\qquad
\sin \frac A2=\sqrt{\frac{1-\cos A}{2}}=\sqrt{\frac{1-\frac35}{2}}=\sqrt{\frac15}=\frac{1}{\sqrt5}.
]

So
[
D=\left(\frac{2}{\sqrt5},\frac{1}{\sqrt5}\right).
]
Let (s=\frac{1}{\sqrt5}). Then (D=(2s,s)).

---

### 2) Use the ratio (\dfrac{BD}{CD}=\dfrac32)

Compute squared lengths (to avoid square roots).

[
BD^2=(x-2s)^2+(0-s)^2=x^2-4sx+5s^2=x^2-4sx+1
]
since (5s^2=1).

For (CD^2):
[
CD^2=\left(\frac{3y}{5}-2s\right)^2+\left(\frac{4y}{5}-s\right)^2
= \left(\frac{9y^2}{25}-\frac{12sy}{5}+4s^2\right)+\left(\frac{16y^2}{25}-\frac{8sy}{5}+s^2\right)
]
[
= y^2-4sy+5s^2=y^2-4sy+1.
]

Thus
[
\frac{x^2-4sx+1}{y^2-4sy+1}=\frac{9}{4}. \tag{R}
]

---

### 3) Use the right angle (\angle BDC=90^\circ)

Orthogonality at (D) gives ((B-D)\cdot(C-D)=0).

[
B-D=(x-2s,-s),\quad C-D=\left(\frac{3y}{5}-2s,\frac{4y}{5}-s\right).
]

Dot product:
[
(x-2s)\left(\frac{3y}{5}-2s\right)+(-s)\left(\frac{4y}{5}-s\right)=0.
]
Expand and simplify:
[
\frac{3xy}{5}-2s(x+y)+1=0. \tag{P}
]

---

### 4) Remove radicals via scaling

Define
[
X=\frac{x}{s}=x\sqrt5,\qquad Y=\frac{y}{s}=y\sqrt5.
]

Rewrite (P) by dividing by (s^2) (note (1/s^2=5)):

[
\frac{3}{5}XY-2(X+Y)+5=0
\ \Longrightarrow\
3XY-10(X+Y)+25=0. \tag{P'}
]

Rewrite (R) by dividing numerator/denominator by (s^2):

[
\frac{X^2-4X+5}{Y^2-4Y+5}=\frac94. \tag{R'}
]

From (P'):
[
3XY-10X-10Y+25=0
\quad\Longrightarrow\quad
X(3Y-10)=10Y-25
\quad\Longrightarrow\quad
X=\frac{10Y-25}{3Y-10}. \tag{*}
]

Substitute (*) into (R'). After simplification, the condition becomes
[
(9Y-20)(9Y-40)(Y^2-4Y+5)=0.
]
The quadratic factor has no real roots, so
[
Y=\frac{20}{9}\quad\text{or}\quad Y=\frac{40}{9}.
]
Then from (*):

* If (Y=\frac{20}{9}), (X=\frac56).
* If (Y=\frac{40}{9}), (X=\frac{35}{6}).

---

### 5) Enforce “(D) is outside (\triangle ABC)”

Let (E=AD\cap BC). Since (D) lies on the internal bisector ray, (D) is outside (\triangle ABC) iff the bisector segment length (AE<AD=1).

Angle-bisector length:
[
AE=\frac{2xy\cos(A/2)}{x+y}.
]
We have (\cos(A/2)=\frac{2}{\sqrt5}=2s).

**Case (X=\frac56, Y=\frac{20}{9}):**
[
x+y=s(X+Y)=s\left(\frac56+\frac{20}{9}\right)=s\cdot\frac{55}{18},
]
[
xy=s^2XY=\frac15\cdot\frac56\cdot\frac{20}{9}=\frac{10}{27}.
]
So
[
AE=\frac{2\cdot\frac{10}{27}\cdot 2s}{s\cdot\frac{55}{18}}
=\frac{16}{33}<1,
]
valid.

The other case gives (AE>1) (so (D) would lie inside), invalid.

Thus ((X,Y)=\left(\frac56,\frac{20}{9}\right)).

---

### 6) Compute (AB+AC)

[
AB+AC=x+y=s(X+Y)=\frac{1}{\sqrt5}\cdot\frac{55}{18}
=\frac{55}{18\sqrt5}
=\frac{11\sqrt5}{18}.
]

So (a=11,\ b=5,\ c=18), hence (a+b+c=34).

Answer: 34
"""

def calculate_perplexity(question_text: str, solution_text: str, model, tokenizer, device="cuda"):
    """Calculate conditional and unconditional perplexity of a solution.
    
    Args:
        question_text: The question/prompt text
        solution_text: The solution text to evaluate
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        
    Returns:
        Tuple of (ppx_conditional, ppx_unconditional, ppx_diff)
    """
    # Tokenize question and solution separately to properly mask the question in labels
    question_tokens = tokenizer(question_text, return_tensors="pt", add_special_tokens=True)
    solution_tokens = tokenizer("\n\n" + solution_text, return_tensors="pt", add_special_tokens=False)
    
    # Move to device
    question_ids = question_tokens.input_ids.to(device)
    solution_ids = solution_tokens.input_ids.to(device)
    
    # Concatenate for input
    input_ids = torch.cat([question_ids, solution_ids], dim=-1)
    
    # Create labels: mask question part with -100 (ignored in loss), keep solution part
    # This ensures loss is only computed on the solution tokens
    labels = torch.cat([
        torch.full_like(question_ids, -100),
        solution_ids
    ], dim=-1)
    
    # Calculate conditional perplexity (with question context)
    with torch.no_grad():
        outputs_conditional = model(input_ids=input_ids, labels=labels)
        ppx_conditional = torch.exp(outputs_conditional.loss)
    
    # Calculate unconditional perplexity (without question context)
    with torch.no_grad():
        outputs_unconditional = model(input_ids=solution_ids, labels=solution_ids)
        ppx_unconditional = torch.exp(outputs_unconditional.loss)
    
    # Perplexity difference (creativity reward metric from LM_R)
    ppx_diff = ppx_unconditional - ppx_conditional
    
    return ppx_conditional.item(), ppx_unconditional.item(), ppx_diff.item()

#%%
base_perplexities = {}

# Evaluate ground truth solution
print("=" * 80)
print("GROUND TRUTH SOLUTION (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, ground_truth_solution, model, tokenizer)
base_perplexities['ground_truth'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

# Evaluate random solution
print("\n" + "=" * 80)
print("RANDOM SOLUTION (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, random_solution, model, tokenizer)
base_perplexities['random'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

# Evaluate another question's solution
print("\n" + "=" * 80)
print("ANOTHER QUESTION'S SOLUTION (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, another_questions_solution, model, tokenizer)
base_perplexities['another_question'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

# Evaluate new solution
print("\n" + "=" * 80)
print("NEW SOLUTION (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, new_solution, model, tokenizer)
base_perplexities['new_solution'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

# Evaluate new solution with ground truth solution
print("\n" + "=" * 80)
print("NEW SOLUTION WITH GROUND TRUTH SOLUTION (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question + "\n\n" + ground_truth_solution, new_solution, model, tokenizer)
base_perplexities['new_solution_with_ground_truth_solution'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

# Evaluate ground truth solution version 2
print("\n" + "=" * 80)
print("GROUND TRUTH SOLUTION VERSION 2 (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, ground_truth_solution_version_2, model, tokenizer)
base_perplexities['ground_truth_version_2'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

# Evaluate ground truth solution version 2 with ground truth solution
print("\n" + "=" * 80)
print("GROUND TRUTH SOLUTION VERSION 2 WITH GROUND TRUTH SOLUTION (Base Model)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question + "\n\n" + ground_truth_solution, ground_truth_solution_version_2, model, tokenizer)
base_perplexities['ground_truth_version_2_with_ground_truth_solution'] = {'cond': ppx_cond, 'uncond': ppx_uncond, 'diff': ppx_diff}
print(f"Conditional perplexity (solution | question): {ppx_cond:.4f}")
print(f"Unconditional perplexity (solution only): {ppx_uncond:.4f}")
print(f"Perplexity difference (unconditional - conditional): {ppx_diff:.4f}")

#%%
"""
Simulate the PEFT fine-tuning process in icrl_lm_r.
"""

from peft import LoraConfig, get_peft_model, TaskType
import copy

#%%
# Create the PEFT model for LM_R (following lm_r_workers.py)
print("\n" + "=" * 80)
print("CREATING PEFT MODEL FOR LM_R")
print("=" * 80)

# LoRA config matching verl/trainer/config/icrl_trainer.yaml
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

# Create PEFT model with LoRA adapter
# IMPORTANT: We need to load a separate base model because get_peft_model modifies the model in-place
# This ensures the original 'model' variable stays unchanged
base_model_for_lm_r = transformers.AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
lm_r_model = get_peft_model(base_model_for_lm_r, peft_config)
lm_r_model.train()

# Create optimizer (matching lm_r_workers.py)
trainable_params = [p for p in lm_r_model.parameters() if p.requires_grad]
base_lr = 5e-4
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=base_lr,
    betas=(0.9, 0.999),
    eps=1e-8,
)

print(f"Created PEFT model with {sum(p.numel() for p in trainable_params):,} trainable parameters")
lm_r_model.print_trainable_parameters()


#%%
# Function to run one step of PEFT fine-tuning (following batch_update_lm_r in lm_r_workers.py)
def finetune_lm_r_step(
    model_to_train,
    optimizer_to_use,
    solution_text: str,
    tokenizer,
    task_reward: float,
    base_lr: float = 5e-4,
    epsilon: float = 0.1,
    device: str = "cuda"
):
    """Run one step of LM_R fine-tuning with LR scaled by task reward.
    
    Args:
        model_to_train: The PEFT model to fine-tune
        optimizer_to_use: The optimizer
        solution_text: The solution text to train on
        tokenizer: Tokenizer
        task_reward: Task reward for LR scaling (higher reward = lower LR)
        base_lr: Base learning rate
        epsilon: Small constant to prevent division by zero
        device: Device to run on
        
    Returns:
        loss value
    """
    # Compute scaled learning rate: lr = base_lr / (epsilon + task_reward)
    # Clamp task_reward to 0 to handle negative rewards
    task_reward = max(0.0, task_reward)
    scaled_lr = base_lr / (epsilon + task_reward)
    
    # Update optimizer learning rate
    for param_group in optimizer_to_use.param_groups:
        param_group['lr'] = scaled_lr
    
    # Tokenize the solution
    solution_tokens = tokenizer(solution_text, return_tensors="pt", add_special_tokens=True)
    input_ids = solution_tokens.input_ids.to(device)
    
    # Single SFT step (following lm_r_workers.py line 606-620)
    optimizer_to_use.zero_grad()
    
    outputs = model_to_train(
        input_ids=input_ids,
        labels=input_ids,  # Train to predict the entire solution
    )
    
    loss = outputs.loss
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), max_norm=1.0)
    
    optimizer_to_use.step()
    
    return loss.item(), scaled_lr

print("\n" + "=" * 80)
print("FINE-TUNING LM_R ON GROUND TRUTH SOLUTION")
print("=" * 80)

# Simulate fine-tuning on ground truth solution with high task reward (correct answer)
task_reward_correct = 0.0  # High reward for correct solution
num_steps = 5  # Run multiple steps

print(f"Task reward: {task_reward_correct}")
print(f"Running {num_steps} fine-tuning steps...")

for step in range(num_steps):
    loss, scaled_lr = finetune_lm_r_step(
        lm_r_model,
        optimizer,
        ground_truth_solution,
        tokenizer,
        task_reward=task_reward_correct,
        base_lr=base_lr,
    )
    print(f"  Step {step+1}: loss={loss:.4f}, scaled_lr={scaled_lr:.6f}")

#%%
def format_with_pct_change(value, base_value, metric_name):
    """Format a value with percentage change from baseline."""
    pct_change = ((value - base_value) / base_value) * 100
    sign = "+" if pct_change > 0 else ""
    return f"{metric_name}: {sign}{pct_change:6.2f}% | {value:.4f} (base: {base_value:.4f})"

# Calculate perplexity differences with the fine-tuned PEFT model
print("\n" + "=" * 80)
print("PERPLEXITY AFTER FINE-TUNING (LM_R)")
print("=" * 80)

# First, verify the original base model is unchanged
print("Verifying original base model is unchanged:")
ppx_cond_check, ppx_uncond_check, ppx_diff_check = calculate_perplexity(
    question, ground_truth_solution, model, tokenizer
)
print(f"  Ground truth with base model: ppx_cond={ppx_cond_check:.4f} (expected: {base_perplexities['ground_truth']['cond']:.4f})")
print(f"  Match: {'✓' if abs(ppx_cond_check - base_perplexities['ground_truth']['cond']) < 0.01 else '✗'}\n")

# Ground truth solution (trained on this) - using fine-tuned LM_R
print("=" * 80)
print("GROUND TRUTH SOLUTION (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, ground_truth_solution, lm_r_model, tokenizer)
base = base_perplexities['ground_truth']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

# Random solution (not trained on this) - using fine-tuned LM_R
print("\n" + "=" * 80)
print("RANDOM SOLUTION (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, random_solution, lm_r_model, tokenizer)
base = base_perplexities['random']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

# Another question's solution (not trained on this) - using fine-tuned LM_R
print("\n" + "=" * 80)
print("ANOTHER QUESTION'S SOLUTION (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, another_questions_solution, lm_r_model, tokenizer)
base = base_perplexities['another_question']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

# New solution (trained on this)
print("\n" + "=" * 80)
print("NEW SOLUTION (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, new_solution, lm_r_model, tokenizer)
base = base_perplexities['new_solution']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

# New solution with ground truth solution (trained on this)
print("\n" + "=" * 80)
print("NEW SOLUTION WITH GROUND TRUTH SOLUTION (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question + "\n\n" + ground_truth_solution, new_solution, lm_r_model, tokenizer)
base = base_perplexities['new_solution_with_ground_truth_solution']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

# Ground truth solution version 2 (trained on this)
print("\n" + "=" * 80)
print("GROUND TRUTH SOLUTION VERSION 2 (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question, ground_truth_solution_version_2, lm_r_model, tokenizer)
base = base_perplexities['ground_truth_version_2']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

# Ground truth solution version 2 with ground truth solution (trained on this)
print("\n" + "=" * 80)
print("GROUND TRUTH SOLUTION VERSION 2 WITH GROUND TRUTH SOLUTION (after fine-tuning)")
print("=" * 80)
ppx_cond, ppx_uncond, ppx_diff = calculate_perplexity(question + "\n\n" + ground_truth_solution, ground_truth_solution_version_2, lm_r_model, tokenizer)
base = base_perplexities['ground_truth_version_2_with_ground_truth_solution']
print(format_with_pct_change(ppx_cond, base['cond'], "Conditional"))
print(format_with_pct_change(ppx_uncond, base['uncond'], "Unconditional"))
print(format_with_pct_change(ppx_diff, base['diff'], "Difference"))

#%% Mutual Information Analysis
def calculate_mutual_information(context, text, model, tokenizer, device="cuda"):
    """Calculate mutual information I(text; context) = H(text) - H(text|context).
    
    Args:
        context: The context/condition (e.g., question)
        text: The text to measure (e.g., solution)
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        
    Returns:
        Mutual information in bits
    """
    ppx_cond, ppx_uncond, _ = calculate_perplexity(context, text, model, tokenizer, device)
    
    # H(text) = log2(PP_uncond(text))
    # H(text|context) = log2(PP_cond(text|context))
    # I(text; context) = H(text) - H(text|context)
    H_text = np.log2(ppx_uncond)
    H_text_given_context = np.log2(ppx_cond)
    mutual_info = H_text - H_text_given_context
    
    return mutual_info

print("\n" + "=" * 80)
print("MUTUAL INFORMATION ANALYSIS (Base Model)")
print("=" * 80)

# Calculate all mutual information quantities
mi_results = {}

# I(s_o, p): MI between another problem's solution and question
print("Calculating I(s_o, p)...")
mi_results['I(s_o, p)'] = {
    'value': calculate_mutual_information(question, another_questions_solution, model, tokenizer),
    'description': 'Another problem\'s solution & question'
}

# I(s_r, p): MI between random text and question
print("Calculating I(s_r, p)...")
mi_results['I(s_r, p)'] = {
    'value': calculate_mutual_information(question, random_solution, model, tokenizer),
    'description': 'Random text & question'
}

# I(s_1, p): MI between solution 1 and question
print("Calculating I(s_1, p)...")
mi_results['I(s_1, p)'] = {
    'value': calculate_mutual_information(question, ground_truth_solution, model, tokenizer),
    'description': 'Solution 1 & question'
}

# I(s_2, p): MI between solution 2 and question
print("Calculating I(s_2, p)...")
mi_results['I(s_2, p)'] = {
    'value': calculate_mutual_information(question, ground_truth_solution_version_2, model, tokenizer),
    'description': 'Solution 2 & question'
}

# I(s_n, p): MI between new approach and question
print("Calculating I(s_n, p)...")
mi_results['I(s_n, p)'] = {
    'value': calculate_mutual_information(question, new_solution, model, tokenizer),
    'description': 'New approach & question'
}

# I(s_1, s_2): MI between solution 1 and solution 2
print("Calculating I(s_1, s_2)...")
mi_results['I(s_1, s_2)'] = {
    'value': calculate_mutual_information(ground_truth_solution, ground_truth_solution_version_2, model, tokenizer),
    'description': 'Solution 1 & solution 2 (similar)'
}

# I(s_1, s_n): MI between solution 1 and new approach
print("Calculating I(s_1, s_n)...")
mi_results['I(s_1, s_n)'] = {
    'value': calculate_mutual_information(ground_truth_solution, new_solution, model, tokenizer),
    'description': 'Solution 1 & new approach (different)'
}

# I(s_1, s_o): MI between solution 1 and another problem's solution
print("Calculating I(s_1, s_o)...")
mi_results['I(s_1, s_o)'] = {
    'value': calculate_mutual_information(ground_truth_solution, another_questions_solution, model, tokenizer),
    'description': 'Solution 1 & another problem\'s solution'
}

# Create DataFrame for display
mi_df = pd.DataFrame([
    {
        'Pair': pair,
        'Mutual Information (bits)': data['value'],
        'Description': data['description']
    }
    for pair, data in mi_results.items()
])

print("\n" + "=" * 80)
print("RESULTS TABLE")
print("=" * 80)
print(mi_df.to_string(index=False))

print("\n" + "=" * 80)
print("EXPECTED RELATIONSHIPS")
print("=" * 80)
print("Expected: I(s_o, p) ≈ I(s_r, p) > I(s_1, p) ≈ I(s_2, p) ≈ I(s_n, p)")
print("Expected: I(s_1, s_2) > I(s_1, s_n) > I(s_1, s_o)")
print()
print("Relationship 1: Solutions vs Problem")
print(f"  I(s_o, p)  = {mi_results['I(s_o, p)']['value']:.4f}  (another problem's solution & question)")
print(f"  I(s_r, p)  = {mi_results['I(s_r, p)']['value']:.4f}  (random text & question)")
print(f"  I(s_1, p)  = {mi_results['I(s_1, p)']['value']:.4f}  (solution 1 & question)")
print(f"  I(s_2, p)  = {mi_results['I(s_2, p)']['value']:.4f}  (solution 2 & question)")
print(f"  I(s_n, p)  = {mi_results['I(s_n, p)']['value']:.4f}  (new approach & question)")
print()
print("Relationship 2: Solution 1 with Other Solutions")
print(f"  I(s_1, s_2) = {mi_results['I(s_1, s_2)']['value']:.4f}  (solution 1 & solution 2 - similar)")
print(f"  I(s_1, s_n) = {mi_results['I(s_1, s_n)']['value']:.4f}  (solution 1 & new approach - different)")
print(f"  I(s_1, s_o) = {mi_results['I(s_1, s_o)']['value']:.4f}  (solution 1 & another problem's solution)")