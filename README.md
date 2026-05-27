# Modelo presa-depredador de tres especies con inercia poblacional

Este proyecto desarrolla una simulación numérica de un sistema presa-depredador de tres especies, aplicado a una cadena alimenticia formada por:

- Pasto
- Conejos
- Zorros

El objetivo principal es analizar cómo cambian estas poblaciones en el tiempo y comparar diferentes métodos numéricos para resolver el sistema.

## Descripción del proyecto

Inicialmente, el modelo presa-depredador se puede plantear como un sistema de primer orden. Sin embargo, para cumplir con el requisito del taller, se incorporó **inercia poblacional**, lo que permite trabajar con ecuaciones diferenciales de segundo orden.

La inercia poblacional representa que las poblaciones no cambian de manera instantánea, sino que tienen una velocidad y una aceleración de cambio. Por esta razón, el modelo incluye términos de amortiguamiento que ayudan a representar la resistencia del sistema frente a cambios bruscos.

## Modelo utilizado

El sistema modela la interacción entre tres especies:

- El pasto crece de forma natural, pero es consumido por los conejos.
- Los conejos crecen al consumir pasto, pero disminuyen por muerte natural y depredación de los zorros.
- Los zorros crecen al alimentarse de conejos, pero también disminuyen por muerte natural.

Para poder resolver el sistema computacionalmente, el modelo de segundo orden fue reducido a un sistema de primer orden usando el vector:

```python
Y = [P, u, C, v, Z, w]
