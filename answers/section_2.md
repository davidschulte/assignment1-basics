## Section 2

### 2.1. Understanding Unidoce

#### (a)
It represents a null-byte.

### (b)
It is enclosed in quotation marks.

### (c)
If you print it, it disappears.


### 2.1. Unicode Encodings

#### (a)
We do not want to spend too many bytes when most symbolds reoccur very often.

#### (b)
"こんにちは": The individual symbols take up more than one byte each.

#### (c)
"\xff\xff" bceause unicode can't decode "\xff". Also "\x80\x00" because everthing larger or equal to "\x80" can't start an encoding.


