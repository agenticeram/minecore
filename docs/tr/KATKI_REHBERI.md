# MineCore katkı rehberi

Açık kaynaklı bir maden mühendisliği platformu kurmamıza yardım ettiğiniz için
teşekkürler. Maden mühendisleri, jeologlar, jeoistatistikçiler, harita
mühendisleri ve yazılımcılar katkı verebilir. Rust bilmeniz gerekmez:
uygulamanın, komut satırının ve testlerin çoğu Python'dur. En değerli
katkıların bir kısmı hiç kod içermez.

English version: [`CONTRIBUTING.md`](../../CONTRIBUTING.md). İki metin
çelişirse İngilizce metin geçerlidir.

## Nasıl katkı verebilirsiniz

| Tür | Örnek | Etiket |
| --- | --- | --- |
| Kod | Yol haritasından bir özellik grubu, hata düzeltmesi, CLI komutu | `phase:P1` … `phase:P14`, `bug` |
| Doğrulama verisi | Sonucu bilinen çözümlü örnek veya veri seti (ders kitabı, makale, el hesabı); teste dönüştürülür | `validation-data` |
| Yöntem incelemesi | Bir algoritmanın yayımlanmış yönteme uyduğunu kontrol etmek, hatalı varsayımları göstermek | `needs-domain-expert` |
| Deneme | Windows sürümünü kendi (anonimleştirilmiş) verinizle çalıştırıp yanlış sonuçları bildirmek | `bug` |
| Belge ve çeviri | Öğretici, Türkçe/İngilizce arayüz metni incelemesi | `documentation` |

Başlangıç noktaları:

- `good first issue`: küçük, kendi içinde tamamlanan görevler.
- `help wanted`: dış katkının özellikle istendiği büyük görevler.
- `validation-data`: kod yazmadan referans örnek verebilecek mühendisler için.
- [`docs/ROADMAP.md`](../ROADMAP.md): P1–P14 fazları ve yapılma sırası.
- [`docs/KNOWN_LIMITATIONS.md`](../KNOWN_LIMITATIONS.md): MineCore'un henüz
  yapamadıkları. Her satır olası bir katkıdır.

## İş akışı

1. **Bir görev (issue) seçin veya açın.** Her değişiklik bir görevle başlar.
   Yeni mühendislik özelliği için *Feature request*, somut iş için *Task*,
   referans veri için *Validation data* formunu kullanın.
2. **Başlamadan önce yorum yazın.** Görevi aldığınızı ve küçük bir düzeltmeden
   büyükse yaklaşımınızı birkaç satırla yazın (yöntem, kaynak, değişecek
   dosyalar). Bir yönetici onaylar veya öneri yapar. Böylece iki kişi aynı işi
   yapmaz, sonradan baştan yazmak gerekmez.
3. **Fork ve dal açın.** Kendi kopyanızda `main`'den bir dal açın, örneğin
   `feat/surface-intersection`.
4. **Dikey dilim olarak yapın** (aşağıdaki *Bitti tanımı*): hesap motoru,
   komut, arayüz girişi, test ve belge aynı PR'da.
5. **Kalite kontrollerini yerelde çalıştırın** (CONTRIBUTING.md, *Quality
   gates*).
6. **Pull request açın**, görevi bağlayın (`Closes #NN`) ve şablonu doldurun:
   ne değişti, nasıl doğrulandı, sınırlamalar.
7. **CI ve inceleme.** CI aynı kontrolleri Windows ve Linux'ta çalıştırır.
   Yönetici mühendislik yöntemini ve kodu inceler; doğrulamada kullandığınız
   kaynak sorulabilir.
8. **Birleştirme** proje sahibi tarafından yapılır. Her PR tek bir değişiklik
   içersin; küçük PR daha hızlı incelenir.

## Bitti tanımı

Bir özellik ancak şunların hepsi sağlandığında biter:

- **Motor:** hesap, arayüzden bağımsız olarak alan modülünde (Python) veya
  `crates/mc-core` içinde (Rust) çalışır.
- **Komut:** her araç bir komutla da çalışır. Her menü girişinin karşılığı
  olan bir `minecore` komutu vardır; bu komut programın içindeki komut
  satırından ve `minecore-cli.exe`'den de çalışır. Böylece betikler ve yapay
  zekâ ajanları ekran kontrolü olmadan her aracı kullanabilir.
- **Arayüz:** çalışma ekranında aracın bir girişi vardır (menü, pencere veya
  panel).
- **Doğrulanmış:** testler sonucu analitik çözüm, el hesabı, yayımlanmış
  çözümlü örnek veya bağımsız bir uygulamayla karşılaştırır. Katalogda veya
  görevde doğrulama sayıları verilmişse bunlar test olur.
- **Dürüst sınırlar:** yapılmayan kısım `docs/KNOWN_LIMITATIONS.md`'ye
  yazılır. İşlevsiz düğme, sabit sonuç veya taslak algoritma olmaz.

## Mühendislik kuralları (özet)

1. Kanıt, iddiadan önce gelir: çalıştırdığınız testler geçince iş biter.
2. Her sayısal algoritma bağımsız olarak doğrulanır.
3. **Eşik ve mevzuat değerleri kullanıcı girdisidir, varsayılan değildir.**
   Kesme tenörü, şev açısı, güvenlik katsayısı, titreşim sınırı, hava hızı
   sınırı, QA/QC kabul sınırı gibi değerleri kullanıcı girer ve proje ile
   saklanır.
4. Kaynak sınıflaması hiçbir zaman JORC, NI 43-101, PERC vb. kodlara uyumlu
   diye sunulmaz; bu yetkin kişi işidir.
5. **Yalnız açık dosya biçimleri.** Diğer maden yazılımlarının kodu, görselleri
   veya belgelenmemiş dosya biçimleri kopyalanmaz.
6. **Yalnız serbest lisanslar** (MIT, BSD, Apache-2.0 vb.). GPL/AGPL bir araç
   yalnızca kullanıcının kurduğu ayrı bir süreç olarak kullanılabilir; bağlanmaz,
   pakete konmaz.
7. Birim, koordinat, geometri, blok model şeması ve proje dosya biçimi yalnızca
   ADR ile değişir.
8. Hedef platform Windows 11'dir; CI Linux'ta da çalışır.
9. Kod ve belgeler İngilizce; görev ve tartışmalar Türkçe veya İngilizce
   olabilir.

## Yol haritası kodları

Görevler, rakip özellik envanterindeki `E-XX-NN` kodlarını (örneğin `E-CD-02`
yüzey araçları) ve ilgili fazı (`phase:P1`) belirtir. Kod önekleri
CONTRIBUTING.md'deki tabloda listelenmiştir. Başka bir yazılımda gördüğünüz
özelliği anlatırken mühendislik ihtiyacını ve yayımlanmış yöntemi yazın; o
yazılımın arayüzünü tarif etmeyin veya kopyalamayın.

## Lisans

Katkı vererek katkınızın [MIT Lisansı](../../LICENSE) altında yayımlanmasını
kabul etmiş olursunuz.
