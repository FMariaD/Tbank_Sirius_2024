# Tbank_Sirius_2024
Отборочное задание на смену Т-Банк Сириус 2024 по Computer Vision

### Алгоритм ###

**1. Salient Object-Aware Background Generation using Text-Guided Diffusion Models**
Источник: `yahoo-inc/photo-background-generation` HuggingFace

Вначале убирает фон с помощью библиотеки transparent-background и генерируем новый с текстовым промптом `'empty plain one-colour %s wall' % color` где для четных и нечетных номеров картинок я использовала grey и white. Также сохраняем маски объектов, которые выдает transparent-background для будущего.

Это сделано в ноутбуке `auto.ipynb`; 
`images_masks` - папка с масками;
`images_made` - папками с замененным фоном

**2. Улучшение фона**
Однако после применения модели от yahoo получаются страшные лица людей и нечитабельный текст на товаров из-за своей высокочастотной природе и низкого разрешения фотографии. Поэтому я делаю небольшой merge двух картинок `images_made` и `images+images_masks`. Чисто на глаз решила вот так: при таких параметрах лица и надписи +- нормальные и в то же время не появляется четких границ между объектом и сгенерированным фоном. Компромисс.

```
mask = np.where(mask > 170, 255, 0)
        mask = mask / 255
        new_img = Image.fromarray(np.uint8(np.multiply(img_arr, mask)))
    
        tmp_arr = np.uint8(np.multiply(img_arr, mask))
        final_arr = np.where(tmp_arr != 0, 
                             tmp_arr,
                             new_img_arr)
```

Второй пункт сделан в ноутбуке `replace_bg.ipynb`. Из него получаем папку `images_final`. Получилось неидеально.


**3. BLIP** 
Источник: `Salesforce/blip-image-captioning-large` HuggingFace

Модель выполняет задачу Image Captioning, т.е. генерирует текстовое описание к картинке. Ей на вход подаем `images_final`.
Сделано в ноутбуке `BLIP_model.ipynb`. Ответы после модели записаны в папку `short_captions`.

**4. LLM** 
Источник: `Gigachat` Сбер

Дальше мы подаем LLMке на вход короткие описания после BLIP'а и просим её создать описания длиннее и более рекламные.
Выбрала Сбер т.к. в нем есть бесплатные API-токены, только этим выбор обусловлен.
Я написала вот такой пред-промпт:

"You need to write an advertisement text for a product. It should be long enough. Try to convince me to buy this product. Write in english. I will provide you a short description of a photo of this product. But there may be person-models mentioned, don't pay attention to them, FOCUS ONLY ON THE PRODUCT.Here is a short description of its photo: \n"

Это сделано в ноутбуке `Gigachat.ipynb` и ответы записаны в папку `long_captions` (сгенерировалисть только около 250 штук, но суть понятна для остальных).

**5. Итог**

Представленное решение может убирать фон с фотографии, генерировать новый - пустую стену цвета из запроса пользователя. Затем создает два описания: краткое и длинное (рекламное). Использованы модели Диффузионная text-guided, Visual Transformer (BLIP и transparent-background) и LLm.
Прошу прощения, что не сделала полноценные картинки с ответами, где всё вместе. **смотреть нужно следующие файлы:**
1) images - первоначальные изображения
2) images_final - итоговое изображение с заменой фона
3) short_captions - короткое описание товара после BLIP
4) long_captions - длинное описание товара после LLM

Если поможет, я начала файл `creating_csv` для соединения  1)-4) в одно...


