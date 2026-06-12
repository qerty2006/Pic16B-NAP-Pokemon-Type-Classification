#import "@preview/clean-math-paper:0.2.4": *
#import "@preview/simple-plot:0.3.0": plot, func-plot
#import "@preview/pyrunner:0.3.0" as py

#let jish(title: "", authors: (), date: "", stuff) = {
  set page(
    footer: context {
      let page_num = counter(page).at(here()).at(0)
      
      set text(10pt)
      line(length: 100%, stroke: 1pt + gray)
      
      // Logic: Only show authors if page > 1
      if page_num > 1 [
        #authors.map(a => a.name).join(", ")
      ]
      
      h(1fr) 
      counter(page).display()
    }
  )
  show: template.with(
    title: title,
    authors: authors,
    date: date
  )
  
  set heading(numbering: none)
  set math.mat(delim: "[")
  set enum(numbering: "a)")
  stuff
}

// Display can force display mode in inlines


#let st = "s.t."
#let iff = $<==>$
#let real = $bb(R)$
#let nat = $bb(N)$
#let int = $bb(Z)$

#let dy = $d y$
#let dx = $d x$
#let dz = $d z$
#let dt = $d t$
#let del = $partial$

#let Let = "Let"

#let rank = "rank"
#let rankb(A) = $"rank"(bold(#A))$
#let null = "null"
#let nullb(A) = $"null"(bold(#A))$
#let dim = "dim"
#let dimb(A) = $"dim"(bold(#A))$
#let im = "im"
#let imb(A) = $"im"(bold(#A))$
#let ker = "ker"
#let kerb(A) = $"ker"(bold(#A))$

#let newpage = {pagebreak()}
#let innerproduct(x, y) = $lr(angle.l #x, #y angle.r)$
#let irect(x) = $#rect(x, inset: 2% + 5pt)$
#let jrect(x) = $#rect($display(#x)$, inset: 1% + 10pt)$

#let problem_box(
  number, body,
  fill : rgb("#e8f5e9"),
  stroke: 1.5pt + rgb("#042f13") ) = rect(
    fill: fill,      // Light green background
    stroke: stroke, // Dark gray border
    radius: 8pt,               // Rounded corners
    width: 100%,               // Make it span the page width
    inset: 12pt,               // Padding inside the box
    [*#number)*  #body]
  )
#let remarkbox(title,body,
  fill : rgb("#e1faff"),
  stroke: 1.5pt + rgb("#2891ae") ) = rect(
    fill: fill,      // Light green background
    stroke: stroke, // Dark gray border
    radius: 8pt,               // Rounded corners
    width: 100%,               // Make it span the page width
    inset: 12pt,               // Padding inside the box
    [*#title*  #body]
  )

#let limit_def(expression,
  N: $N$,
  epsilon: $epsilon$
) = $(forall #epsilon in real_(>0))(exists #N in real)(forall n in nat)[n >= #N => expression]$

#let enumerate(style, body) = {set enum(numbering: style); body}

#let pm = $plus.minus$
#let proofend = [#h(1fr) #math.qed]
#let implies = $=>$

lol
